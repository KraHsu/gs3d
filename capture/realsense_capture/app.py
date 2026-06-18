"""Entry point for the RealSense capture GUI."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .gui import CaptureWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = CaptureWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
