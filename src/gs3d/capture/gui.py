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

# Cohesive dark theme (does not rely on the OS palette).
APP_QSS = """
QWidget { background: #1e1f22; color: #e4e6eb; font-size: 13px; }
QLabel { background: transparent; }
QGroupBox {
    background: #232529; font-weight: 600;
    border: 1px solid #3a3d42; border-radius: 8px;
    margin-top: 12px; padding: 12px 10px 10px 10px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #9aa0a8;
}
QPushButton {
    background: #313338; color: #e4e6eb; border: 1px solid #3a3d42;
    border-radius: 6px; padding: 6px 14px;
}
QPushButton:hover { background: #3a3d42; }
QPushButton:pressed { background: #2a2c30; }
QPushButton:disabled { color: #6b7078; background: #2a2b2e; border-color: #303236; }
QLineEdit, QSpinBox {
    background: #2b2d31; color: #e4e6eb; border: 1px solid #3a3d42;
    border-radius: 6px; padding: 5px 7px; selection-background-color: #2ea043;
}
QLineEdit:focus, QSpinBox:focus { border-color: #4a87e8; }
QLineEdit:disabled, QSpinBox:disabled { color: #8a8f98; background: #26272b; }
QCheckBox { spacing: 6px; background: transparent; }
QCheckBox::indicator {
    width: 16px; height: 16px; border: 1px solid #4a4d52;
    border-radius: 4px; background: #2b2d31;
}
QCheckBox::indicator:checked { background: #2ea043; border-color: #2ea043; }
QToolTip { background: #26282c; color: #e4e6eb; border: 1px solid #3a3d42; }
"""


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


def _sharpness(bgr: np.ndarray) -> float:
    """Variance of the Laplacian — higher is sharper (same metric as curate_dataset)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


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
        self._dropped = 0

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._snap)
        QShortcut(QKeySequence(Qt.Key.Key_R), self, activated=self._toggle_record)

    # -- UI ----------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setStyleSheet(APP_QSS)
        # Dataset group ----------------------------------------------------
        self.out_edit = QLineEdit(str(self._default_output_dir()))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.name_edit = QLineEdit("dataset01")
        self.name_edit.textChanged.connect(self._refresh_target)
        self.target_label = QLabel()
        self.target_label.setStyleSheet("color:#9aa0a8;")

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
        self.preview.setStyleSheet(
            "background:#141517; color:#777; border:1px solid #3a3d42; border-radius:8px;"
        )

        # Camera group -----------------------------------------------------
        self.cam_btn = QPushButton("Start camera")
        self.cam_btn.clicked.connect(self._toggle_camera)
        self.depth_chk = QCheckBox("Show depth")
        self.imu_chk = QCheckBox("Capture IMU")
        self.imu_chk.setToolTip("Record D435i accel+gyro to imu.jsonl (set before Start camera).")
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)
        self.fps_spin.setToolTip("Preview/capture polling rate (capped by the camera's stream fps).")
        self.fps_spin.valueChanged.connect(self._on_fps)
        self.profile_label = QLabel("")
        self.profile_label.setStyleSheet("color:#9aa0a8;")
        cam_row = QHBoxLayout()
        cam_row.addWidget(self.cam_btn)
        cam_row.addWidget(self.depth_chk)
        cam_row.addWidget(self.imu_chk)
        cam_row.addWidget(QLabel("FPS"))
        cam_row.addWidget(self.fps_spin)
        cam_row.addWidget(self.profile_label, 1)

        # Exposure / gain (shorter exposure => less motion blur, but darker).
        self.ae_chk = QCheckBox("Auto exposure")
        self.ae_chk.setChecked(True)
        self.ae_chk.toggled.connect(self._on_ae_toggled)
        self.exp_spin = QSpinBox()
        self.exp_spin.setSuffix(" us")
        self.exp_spin.setKeyboardTracking(False)  # apply on Enter / focus-out, not per keystroke
        self.exp_spin.valueChanged.connect(self._on_exposure)
        self.gain_spin = QSpinBox()
        self.gain_spin.setKeyboardTracking(False)
        self.gain_spin.valueChanged.connect(self._on_gain)
        exp_row = QHBoxLayout()
        exp_row.addWidget(self.ae_chk)
        exp_row.addSpacing(12)
        exp_row.addWidget(QLabel("Exposure"))
        exp_row.addWidget(self.exp_spin)
        exp_row.addSpacing(12)
        exp_row.addWidget(QLabel("Gain"))
        exp_row.addWidget(self.gain_spin)
        exp_row.addStretch(1)

        cam_box = QGroupBox("Camera")
        cam_lay = QVBoxLayout(cam_box)
        cam_lay.addLayout(cam_row)
        cam_lay.addLayout(exp_row)

        # Recording group --------------------------------------------------
        self.rec_btn = QPushButton("●  Record")
        self.rec_btn.setObjectName("recordBtn")
        self.rec_btn.setMinimumWidth(120)
        self.rec_btn.clicked.connect(self._toggle_record)

        self.snap_btn = QPushButton("Snap (Space)")
        self.snap_btn.clicked.connect(self._snap)

        self.every_spin = QSpinBox()
        self.every_spin.setRange(1, 30)
        self.every_spin.setValue(3)
        self.every_spin.setToolTip("Save one frame out of every N previewed frames while recording.")

        self.rec_indicator = QLabel("")
        self.rec_indicator.setStyleSheet("color:#ff5c5c; font-weight:bold;")
        self.stats_label = QLabel("Frames: 0")
        self.stats_label.setStyleSheet("font-weight:bold;")

        rec_row = QHBoxLayout()
        rec_row.setSpacing(8)
        rec_row.addWidget(self.rec_btn)
        rec_row.addWidget(self.snap_btn)
        rec_row.addSpacing(12)
        rec_row.addWidget(QLabel("capture every"))
        rec_row.addWidget(self.every_spin)
        rec_row.addWidget(QLabel("frames"))
        rec_row.addStretch(1)
        rec_row.addWidget(self.rec_indicator)
        rec_row.addWidget(self.stats_label)
        # Sharpness gate: skip blurry frames while recording.
        self.gate_chk = QCheckBox("Sharpness gate")
        self.gate_chk.setToolTip("While recording, skip frames blurrier than the threshold.")
        self.gate_spin = QSpinBox()
        self.gate_spin.setRange(0, 1000)
        self.gate_spin.setValue(50)
        self.sharp_label = QLabel("sharp —")
        self.sharp_label.setStyleSheet("color:#9aa0a8;")
        self.dropped_label = QLabel("dropped: 0")
        self.dropped_label.setStyleSheet("color:#9aa0a8;")
        gate_row = QHBoxLayout()
        gate_row.addWidget(self.gate_chk)
        gate_row.addWidget(QLabel("min sharpness"))
        gate_row.addWidget(self.gate_spin)
        gate_row.addStretch(1)
        gate_row.addWidget(self.sharp_label)
        gate_row.addSpacing(12)
        gate_row.addWidget(self.dropped_label)

        rec_box = QGroupBox("Recording")
        rec_lay = QVBoxLayout(rec_box)
        rec_lay.addLayout(rec_row)
        rec_lay.addLayout(gate_row)

        self.status = QLabel("Idle.")
        tips = QLabel(TIPS)
        tips.setWordWrap(True)
        tips.setStyleSheet("color:#8a8f98;")

        root = QVBoxLayout(self)
        root.addWidget(ds_box)
        root.addWidget(self.preview, 1)
        root.addWidget(cam_box)
        root.addWidget(rec_box)
        root.addWidget(self.status)
        root.addWidget(tips)

        self._style_record(False)
        self._set_capture_enabled(False)
        self._refresh_target()

    def _style_record(self, recording: bool) -> None:
        if recording:
            self.rec_btn.setText("■  Stop")
            self.rec_btn.setStyleSheet(
                "#recordBtn{background:#da3633;color:white;border:1px solid #da3633;"
                "border-radius:6px;padding:6px 14px;font-weight:700;}"
                "#recordBtn:hover{background:#c52f2c;}"
            )
        else:
            self.rec_btn.setText("●  Record")
            self.rec_btn.setStyleSheet(
                "#recordBtn{color:#3fb950;background:#1d2a20;border:1px solid #2ea043;"
                "border-radius:6px;padding:6px 14px;font-weight:700;}"
                "#recordBtn:hover{background:#22331f;}"
                "#recordBtn:disabled{color:#6b7078;border-color:#303236;background:#2a2b2e;}"
            )

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
        self._init_exposure_controls()
        self._set_capture_enabled(True)
        self._missed = 0
        self.status.setText("Camera streaming. Press ● Record to capture a dataset.")
        self.timer.start(self._timer_interval_ms())
        return True

    def _timer_interval_ms(self) -> int:
        return max(1, round(1000 / self.fps_spin.value()))

    def _on_fps(self, _val: int) -> None:
        if self.timer.isActive():
            self.timer.setInterval(self._timer_interval_ms())

    # -- exposure / gain ---------------------------------------------------
    def _init_exposure_controls(self) -> None:
        cam = self.camera
        has_exp = cam.supports_exposure()
        self.ae_chk.setEnabled(has_exp)
        if has_exp:
            lo, hi, dflt = cam.exposure_range()
            self.exp_spin.blockSignals(True)
            self.exp_spin.setRange(int(lo), int(hi))
            self.exp_spin.setValue(int(dflt))
            self.exp_spin.blockSignals(False)
        if cam.supports_gain():
            glo, ghi, gd = cam.gain_range()
            self.gain_spin.blockSignals(True)
            self.gain_spin.setRange(int(glo), int(ghi))
            self.gain_spin.setValue(int(gd))
            self.gain_spin.blockSignals(False)
        # Default to auto-exposure on; the inputs take over when it is unchecked.
        self.ae_chk.setChecked(True)
        cam.set_auto_exposure(True)
        self.exp_spin.setEnabled(False)
        self.gain_spin.setEnabled(False)

    def _on_ae_toggled(self, on: bool) -> None:
        if self.camera is not None:
            self.camera.set_auto_exposure(on)
            if not on:  # push current input values as the manual setpoint
                self.camera.set_exposure(self.exp_spin.value())
                self.camera.set_gain(self.gain_spin.value())
        self.exp_spin.setEnabled(not on)
        self.gain_spin.setEnabled(not on)

    def _on_exposure(self, val: int) -> None:
        if self.camera is not None and not self.ae_chk.isChecked():
            self.camera.set_exposure(val)

    def _on_gain(self, val: int) -> None:
        if self.camera is not None and not self.ae_chk.isChecked():
            self.camera.set_gain(val)

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
        self._dropped = 0
        self.dropped_label.setText("dropped: 0")
        self._style_record(True)
        # Lock dataset identity while recording.
        self.name_edit.setEnabled(False)
        self.out_edit.setEnabled(False)
        self.status.setText(f"Recording → {self.writer.scene_dir}")

    def _stop_record(self) -> None:
        self._recording = False
        if self.writer is not None:
            self.writer.flush_meta()
        self._style_record(False)
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

        sharp = _sharpness(frame.color_bgr)
        self.sharp_label.setText(f"sharp {sharp:.0f}")

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
                if self.gate_chk.isChecked() and sharp < self.gate_spin.value():
                    self._dropped += 1
                    self.dropped_label.setText(f"dropped: {self._dropped}")
                else:
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
