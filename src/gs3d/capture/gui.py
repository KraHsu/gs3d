"""PySide6 GUI for recording a D435i dataset for 3D Gaussian Splatting.

Live RGB (and optional depth) preview with a one-click **Record** toggle that
continuously captures frames into a named dataset (COLMAP/gsplat layout). Single
**Snap** shots are also available. Output per dataset:

    data/<dataset>/  images/  depth/  intrinsics.json  meta.json
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .camera import CameraError, FrameTimeout, RealSenseCamera
from .writer import DatasetWriter

TIPS = (
    "Recording tip: press ● Record, then orbit the subject slowly (one or two loops), "
    "keeping ~70% overlap. Prefer a textured, well-lit, static scene. ~100–300 frames is plenty."
)


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    rgb = np.ascontiguousarray(rgb)
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _depth_to_qimage(depth_mm: np.ndarray) -> QImage:
    # Normalize for display only (visualization, not stored).
    depth = depth_mm.astype(np.float32)
    valid = depth[depth > 0]
    hi = float(np.percentile(valid, 95)) if valid.size else 1.0
    norm = np.clip(depth / max(hi, 1.0), 0.0, 1.0)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return _bgr_to_qimage(colored)


class CaptureWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RealSense D435i — 3DGS Dataset Recorder")
        self.resize(1180, 820)

        self.camera: RealSenseCamera | None = None
        self.writer: DatasetWriter | None = None
        self._last_frame = None
        self._recording = False
        self._rec_t0 = 0.0
        self._frames_since_cap = 0
        self._missed = 0

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._snap)
        QShortcut(QKeySequence(Qt.Key.Key_R), self, activated=self._toggle_record)

    # -- UI ----------------------------------------------------------------
    def _build_ui(self) -> None:
        # Dataset group ----------------------------------------------------
        self.out_edit = QLineEdit(str(self._default_output_dir()))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.name_edit = QLineEdit("dataset01")
        self.name_edit.textChanged.connect(self._refresh_target)
        self.target_label = QLabel()
        self.target_label.setStyleSheet("color:#888;")

        ds_form = QHBoxLayout()
        ds_form.addWidget(QLabel("Output:"))
        ds_form.addWidget(self.out_edit, 1)
        ds_form.addWidget(browse)
        ds_form.addWidget(QLabel("Dataset:"))
        ds_form.addWidget(self.name_edit)
        ds_box = QGroupBox("Dataset")
        ds_lay = QVBoxLayout(ds_box)
        ds_lay.addLayout(ds_form)
        ds_lay.addWidget(self.target_label)

        # Preview ----------------------------------------------------------
        self.preview = QLabel("Preview — press Start camera")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(960, 540)
        self.preview.setStyleSheet("background:#111; color:#888;")

        # Camera controls --------------------------------------------------
        self.cam_btn = QPushButton("Start camera")
        self.cam_btn.clicked.connect(self._toggle_camera)
        self.depth_chk = QCheckBox("Show depth")
        self.imu_chk = QCheckBox("Capture IMU")
        self.imu_chk.setToolTip("Record D435i accel+gyro to imu.jsonl (set before Start camera).")
        self.profile_label = QLabel("")
        self.profile_label.setStyleSheet("color:#888;")
        cam_row = QHBoxLayout()
        cam_row.addWidget(self.cam_btn)
        cam_row.addWidget(self.depth_chk)
        cam_row.addWidget(self.imu_chk)
        cam_row.addWidget(self.profile_label, 1)

        # Recording group --------------------------------------------------
        self.rec_btn = QPushButton("●  Record")
        self.rec_btn.setMinimumHeight(44)
        self.rec_btn.setStyleSheet("font-size:16px; font-weight:bold;")
        self.rec_btn.clicked.connect(self._toggle_record)

        self.snap_btn = QPushButton("Snap (Space)")
        self.snap_btn.clicked.connect(self._snap)

        self.every_spin = QSpinBox()
        self.every_spin.setRange(1, 30)
        self.every_spin.setValue(3)
        self.every_spin.setToolTip("Save one frame out of every N previewed frames while recording.")

        self.rec_indicator = QLabel("")
        self.rec_indicator.setStyleSheet("color:#e33; font-weight:bold;")
        self.stats_label = QLabel("Frames: 0")
        self.stats_label.setStyleSheet("font-weight:bold;")

        rec_row = QHBoxLayout()
        rec_row.addWidget(self.rec_btn, 2)
        rec_row.addWidget(self.snap_btn, 1)
        rec_row.addWidget(QLabel("capture every"))
        rec_row.addWidget(self.every_spin)
        rec_row.addWidget(QLabel("frames"))
        rec_row.addStretch(1)
        rec_row.addWidget(self.rec_indicator)
        rec_row.addWidget(self.stats_label)
        rec_box = QGroupBox("Recording")
        rec_lay = QVBoxLayout(rec_box)
        rec_lay.addLayout(rec_row)

        self.status = QLabel("Idle.")
        tips = QLabel(TIPS)
        tips.setWordWrap(True)
        tips.setStyleSheet("color:#666;")

        root = QVBoxLayout(self)
        root.addWidget(ds_box)
        root.addWidget(self.preview, 1)
        root.addLayout(cam_row)
        root.addWidget(rec_box)
        root.addWidget(self.status)
        root.addWidget(tips)

        self._set_capture_enabled(False)
        self._refresh_target()

    @staticmethod
    def _default_output_dir() -> Path:
        # Walk up to the project root (the dir containing pyproject.toml), use its data/.
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "pyproject.toml").exists():
                return parent / "data"
        return Path.cwd() / "data"

    def _set_capture_enabled(self, on: bool) -> None:
        self.rec_btn.setEnabled(on)
        self.snap_btn.setEnabled(on)

    def _refresh_target(self) -> None:
        name = self.name_edit.text().strip() or "dataset01"
        target = Path(self.out_edit.text()).expanduser() / name
        existing = len(list((target / "images").glob("*.jpg"))) if target.exists() else 0
        note = f"  (exists: {existing} frames — recording appends)" if existing else "  (new)"
        self.target_label.setText(f"→ {target}{note}")

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Output folder", self.out_edit.text())
        if chosen:
            self.out_edit.setText(chosen)
            self._refresh_target()

    # -- camera ------------------------------------------------------------
    def _toggle_camera(self) -> None:
        if self.camera is None:
            self._start_camera()
        else:
            self._stop_camera()

    def _start_camera(self) -> bool:
        want_imu = self.imu_chk.isChecked()
        try:
            self.camera = RealSenseCamera(enable_imu=want_imu)
            self.camera.start()
        except CameraError as exc:
            self.camera = None
            QMessageBox.critical(self, "Camera error", str(exc))
            return False
        self.cam_btn.setText("Stop camera")
        imu_note = ""
        if want_imu:
            imu_note = "  +IMU" if self.camera.imu_enabled else "  (IMU unavailable)"
        self.profile_label.setText(self.camera.active_profile + imu_note)
        self.imu_chk.setEnabled(False)
        self._set_capture_enabled(True)
        self._missed = 0
        self.status.setText("Camera streaming. Press ● Record to capture a dataset.")
        self.timer.start(30)
        return True

    def _stop_camera(self) -> None:
        if self._recording:
            self._stop_record()
        self.timer.stop()
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        self.cam_btn.setText("Start camera")
        self.profile_label.setText("")
        self.imu_chk.setEnabled(True)
        self.preview.setText("Preview — press Start camera")
        self._set_capture_enabled(False)
        self.status.setText("Camera stopped.")

    # -- dataset writer ----------------------------------------------------
    def _ensure_writer(self) -> bool:
        """Create (or switch to) the writer for the current dataset name."""
        if self.camera is None or self.camera.color_intrinsics is None:
            return False
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Dataset", "Please enter a dataset name.")
            return False
        out_root = Path(self.out_edit.text()).expanduser()
        if self.writer is None or self.writer.scene_dir.name != name:
            try:
                self.writer = DatasetWriter(out_root, name, self.camera.color_intrinsics)
            except OSError as exc:
                QMessageBox.critical(self, "Dataset", f"Cannot create dataset folder:\n{exc}")
                return False
        return True

    # -- recording ---------------------------------------------------------
    def _toggle_record(self) -> None:
        if self._recording:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self) -> None:
        if self.camera is None and not self._start_camera():
            return
        if not self._ensure_writer():
            return
        self._recording = True
        self._rec_t0 = time.monotonic()
        self._frames_since_cap = 0
        self.rec_btn.setText("■  Stop")
        self.rec_btn.setStyleSheet("font-size:16px; font-weight:bold; background:#e33; color:white;")
        # Lock dataset identity while recording.
        self.name_edit.setEnabled(False)
        self.out_edit.setEnabled(False)
        self.status.setText(f"Recording → {self.writer.scene_dir}")

    def _stop_record(self) -> None:
        self._recording = False
        if self.writer is not None:
            self.writer.flush_meta()
        self.rec_btn.setText("●  Record")
        self.rec_btn.setStyleSheet("font-size:16px; font-weight:bold;")
        self.rec_indicator.setText("")
        self.name_edit.setEnabled(True)
        self.out_edit.setEnabled(True)
        n = self.writer.count if self.writer else 0
        self.status.setText(f"Stopped. {n} frames saved (meta.json written).")

    # -- per-frame loop ----------------------------------------------------
    def _on_tick(self) -> None:
        if self.camera is None:
            return
        try:
            frame = self.camera.wait_for_frame(timeout_ms=2000)
        except FrameTimeout:
            self._missed += 1
            self.status.setText(f"Waiting for frames… (missed {self._missed})")
            return
        except CameraError as exc:
            self.status.setText(f"Frame error: {exc}")
            return
        self._missed = 0
        self._last_frame = frame

        # Drain IMU each tick (bounds the buffer); persist only while recording.
        if self.camera.imu_enabled:
            samples = self.camera.drain_imu()
            if self._recording and self.writer is not None:
                self.writer.append_imu(samples)

        qimg = _depth_to_qimage(frame.depth_mm) if self.depth_chk.isChecked() else _bgr_to_qimage(frame.color_bgr)
        self.preview.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        if self._recording:
            self._frames_since_cap += 1
            if self._frames_since_cap >= self.every_spin.value():
                self._frames_since_cap = 0
                self.writer.save(frame)
            elapsed = time.monotonic() - self._rec_t0
            # Blink the REC dot ~1 Hz.
            self.rec_indicator.setText("● REC" if int(elapsed * 2) % 2 == 0 else "   REC")
            self.stats_label.setText(f"Frames: {self.writer.count}  |  {elapsed:5.1f}s")

    def _snap(self) -> None:
        if self.camera is None or self._last_frame is None:
            return
        if not self._ensure_writer():
            return
        stem = self.writer.save(self._last_frame)
        self.stats_label.setText(f"Frames: {self.writer.count}")
        self.status.setText(f"Snapped {stem}.jpg → {self.writer.scene_dir.name}")

    # -- shutdown ----------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        self._stop_camera()
        super().closeEvent(event)
