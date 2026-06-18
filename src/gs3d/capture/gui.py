"""PySide6 GUI for capturing a D435i image set.

Live RGB (and optional depth) preview, manual + interval capture, frame counter,
and a COLMAP/gsplat dataset layout written under the chosen output folder.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
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
    "Tips: orbit the subject slowly, keep ~70% overlap between shots, avoid motion blur, "
    "and prefer a textured, well-lit, static scene. Aim for 80–200 images."
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
        self.setWindowTitle("RealSense D435i — 3DGS Capture")
        self.resize(1180, 760)

        self.camera: RealSenseCamera | None = None
        self.writer: DatasetWriter | None = None
        self._frames_since_auto = 0
        self._last_frame = None
        self._missed = 0

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._snap)

    # -- UI ----------------------------------------------------------------
    def _build_ui(self) -> None:
        self.preview = QLabel("Preview — press Start stream")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(960, 540)
        self.preview.setStyleSheet("background:#111; color:#888;")

        self.out_edit = QLineEdit(str(self._default_output_dir()))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)

        self.scene_edit = QLineEdit("scene01")

        self.start_btn = QPushButton("Start stream")
        self.start_btn.clicked.connect(self._toggle_stream)

        self.snap_btn = QPushButton("Snap (Space)")
        self.snap_btn.clicked.connect(self._snap)
        self.snap_btn.setEnabled(False)

        self.depth_chk = QCheckBox("Show depth")

        self.auto_chk = QCheckBox("Auto-capture every")
        self.auto_spin = QSpinBox()
        self.auto_spin.setRange(1, 120)
        self.auto_spin.setValue(10)
        self.auto_unit = QLabel("frames")

        self.count_label = QLabel("Saved: 0")
        self.count_label.setStyleSheet("font-weight:bold;")
        self.status = QLabel("Idle.")

        # Layout
        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("Output:"))
        row_out.addWidget(self.out_edit, 1)
        row_out.addWidget(browse)
        row_out.addWidget(QLabel("Scene:"))
        row_out.addWidget(self.scene_edit)

        row_ctrl = QHBoxLayout()
        row_ctrl.addWidget(self.start_btn)
        row_ctrl.addWidget(self.snap_btn)
        row_ctrl.addWidget(self.depth_chk)
        row_ctrl.addStretch(1)
        row_ctrl.addWidget(self.auto_chk)
        row_ctrl.addWidget(self.auto_spin)
        row_ctrl.addWidget(self.auto_unit)
        row_ctrl.addStretch(1)
        row_ctrl.addWidget(self.count_label)

        root = QVBoxLayout(self)
        root.addLayout(row_out)
        root.addWidget(self.preview, 1)
        root.addLayout(row_ctrl)
        root.addWidget(self.status)
        tips = QLabel(TIPS)
        tips.setWordWrap(True)
        tips.setStyleSheet("color:#666;")
        root.addWidget(tips)

    @staticmethod
    def _default_output_dir() -> Path:
        # Walk up to the project root (the dir containing pyproject.toml), use its data/.
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "pyproject.toml").exists():
                return parent / "data"
        return Path.cwd() / "data"

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Output folder", self.out_edit.text())
        if chosen:
            self.out_edit.setText(chosen)

    # -- stream control ----------------------------------------------------
    def _toggle_stream(self) -> None:
        if self.camera is None:
            self._start_stream()
        else:
            self._stop_stream()

    def _start_stream(self) -> None:
        scene = self.scene_edit.text().strip()
        if not scene:
            QMessageBox.warning(self, "Scene", "Please enter a scene name.")
            return
        try:
            self.camera = RealSenseCamera()
            self.camera.start()
            assert self.camera.color_intrinsics is not None
            out_root = Path(self.out_edit.text()).expanduser()
            self.writer = DatasetWriter(out_root, scene, self.camera.color_intrinsics)
        except CameraError as exc:
            self.camera = None
            QMessageBox.critical(self, "Camera error", str(exc))
            return

        self.count_label.setText(f"Saved: {self.writer.count}")
        self.start_btn.setText("Stop stream")
        self.snap_btn.setEnabled(True)
        self.scene_edit.setEnabled(False)
        self.out_edit.setEnabled(False)
        self.status.setText(f"Streaming [{self.camera.active_profile}] → {self.writer.scene_dir}")
        self._frames_since_auto = 0
        self.timer.start(30)

    def _stop_stream(self) -> None:
        self.timer.stop()
        if self.writer is not None:
            self.writer.flush_meta()
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        self.start_btn.setText("Start stream")
        self.snap_btn.setEnabled(False)
        self.scene_edit.setEnabled(True)
        self.out_edit.setEnabled(True)
        self.status.setText("Stopped. meta.json written.")

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

        if self.depth_chk.isChecked():
            qimg = _depth_to_qimage(frame.depth_mm)
        else:
            qimg = _bgr_to_qimage(frame.color_bgr)
        self.preview.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        if self.auto_chk.isChecked():
            self._frames_since_auto += 1
            if self._frames_since_auto >= self.auto_spin.value():
                self._frames_since_auto = 0
                self._snap()

    def _snap(self) -> None:
        if self.writer is None or self._last_frame is None:
            return
        stem = self.writer.save(self._last_frame)
        self.count_label.setText(f"Saved: {self.writer.count}")
        self.status.setText(f"Captured {stem}.jpg")

    # -- shutdown ----------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        self._stop_stream()
        super().closeEvent(event)
