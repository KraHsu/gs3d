"""Thin wrapper around pyrealsense2 for the D435i.

Streams color + depth, aligns depth to the color frame, and exposes the color
camera intrinsics (needed downstream for reference / depth back-projection).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - import guard for clearer error
    raise ImportError(
        "pyrealsense2 is not installed. Run `uv sync` inside the `capture/` project "
        "on a Windows machine with the Intel RealSense SDK / drivers available."
    ) from exc


@dataclass
class Intrinsics:
    """Pinhole intrinsics of the (color) stream, in pixels."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str
    coeffs: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Frame:
    """A single aligned capture."""

    color_bgr: np.ndarray  # HxWx3 uint8, BGR (OpenCV convention)
    depth_mm: np.ndarray  # HxW uint16, millimetres, aligned to color
    depth_scale: float  # metres per depth unit (depth_mm * scale = metres)


# (color (w,h), depth (w,h), fps) candidates tried in order. The first that the
# device/USB link can resolve is used. USB3 unlocks the high-res entries; USB2.1
# typically falls back to 640x480@30.
DEFAULT_PROFILES: list[tuple[tuple[int, int], tuple[int, int], int]] = [
    ((1280, 720), (848, 480), 30),
    ((1280, 720), (640, 480), 15),
    ((640, 480), (640, 480), 30),
    ((640, 480), (640, 480), 15),
    ((424, 240), (480, 270), 30),
]


class CameraError(RuntimeError):
    pass


class RealSenseCamera:
    """Context-managed RealSense pipeline.

    Example
    -------
    >>> with RealSenseCamera() as cam:
    ...     frame = cam.wait_for_frame()
    ...     intr = cam.color_intrinsics
    """

    def __init__(
        self,
        profiles: list[tuple[tuple[int, int], tuple[int, int], int]] | None = None,
    ) -> None:
        self.profiles = profiles or DEFAULT_PROFILES

        self._pipeline: "rs.pipeline | None" = None
        self._align: "rs.align | None" = None
        self._depth_scale: float = 0.001  # D435i default: 1 unit = 1 mm
        self.color_intrinsics: Intrinsics | None = None
        self.active_profile: str = ""

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._pipeline is not None:
            return

        ctx = rs.context()
        if len(ctx.query_devices()) == 0:
            raise CameraError(
                "No RealSense device detected. Check the USB connection and drivers."
            )

        pipeline = rs.pipeline()
        profile = None
        errors = []
        for (cw, ch), (dw, dh), fps in self.profiles:
            config = rs.config()
            config.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, fps)
            try:
                profile = pipeline.start(config)
                self.active_profile = f"color {cw}x{ch} + depth {dw}x{dh} @ {fps}fps"
                break
            except RuntimeError as exc:  # this combo isn't resolvable on the link
                errors.append(f"  {cw}x{ch}/{dw}x{dh}@{fps}: {exc}")

        if profile is None:
            raise CameraError(
                "Could not start any stream profile (USB bandwidth / driver issue).\n"
                + "\n".join(errors)
            )

        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())

        # Align depth into the color frame so they share intrinsics + resolution.
        self._align = rs.align(rs.stream.color)

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        i = color_profile.get_intrinsics()
        self.color_intrinsics = Intrinsics(
            width=i.width,
            height=i.height,
            fx=float(i.fx),
            fy=float(i.fy),
            cx=float(i.ppx),
            cy=float(i.ppy),
            model=str(i.model),
            coeffs=[float(c) for c in i.coeffs],
        )
        self._pipeline = pipeline

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None
                self._align = None

    def __enter__(self) -> "RealSenseCamera":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- capture -----------------------------------------------------------
    def wait_for_frame(self, timeout_ms: int = 5000) -> Frame:
        if self._pipeline is None or self._align is None:
            raise CameraError("Camera is not started.")

        frames = self._pipeline.wait_for_frames(timeout_ms)
        frames = self._align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise CameraError("Incomplete frame received from camera.")

        color_bgr = np.asanyarray(color.get_data())
        depth_mm = np.asanyarray(depth.get_data()).astype(np.uint16)
        return Frame(color_bgr=color_bgr, depth_mm=depth_mm, depth_scale=self._depth_scale)

    @property
    def is_running(self) -> bool:
        return self._pipeline is not None


def list_devices() -> list[str]:
    """Return human-readable names of connected RealSense devices."""
    ctx = rs.context()
    return [
        f"{d.get_info(rs.camera_info.name)} (SN {d.get_info(rs.camera_info.serial_number)})"
        for d in ctx.query_devices()
    ]
