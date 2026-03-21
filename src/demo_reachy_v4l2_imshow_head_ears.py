import os
import socket
import time
from pathlib import Path

import cv2
import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


def _wait_for_tcp(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: OSError | None = None

    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)

    raise ConnectionError(
        f"Reachy Mini daemon not reachable at {host}:{port} after {timeout_s:.1f}s. "
        f"Last error: {last_error}"
    )


def _iter_v4l2_devices() -> list[tuple[str, str]]:
    devices: list[tuple[str, str]] = []
    for name_path in sorted(Path("/sys/class/video4linux").glob("video*/name")):
        video = name_path.parent.name  # video4
        dev = f"/dev/{video}"
        try:
            name = name_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        devices.append((dev, name))
    return devices


def _open_reachy_camera() -> cv2.VideoCapture:
    def dev_to_index(dev: str) -> int | None:
        # '/dev/video4' -> 4
        base = Path(dev).name
        if base.startswith("video") and base[5:].isdigit():
            return int(base[5:])
        return None

    # Allow forcing a device (e.g. /dev/video4) or integer index.
    forced = os.getenv("REACHY_CAMERA_DEVICE", "").strip()
    if forced:
        if forced.isdigit():
            cap = cv2.VideoCapture(int(forced), cv2.CAP_V4L2)
        else:
            idx = dev_to_index(forced)
            if idx is None:
                raise RuntimeError(
                    "REACHY_CAMERA_DEVICE must be an integer (e.g. '4') or a /dev/videoX path. "
                    f"Got: {forced}"
                )
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open REACHY_CAMERA_DEVICE={forced}")
        return cap

    warmup_s = float(os.getenv("REACHY_CAMERA_WARMUP", "3"))
    devices = _iter_v4l2_devices()

    candidates = [
        (dev, name)
        for (dev, name) in devices
        if "reachy" in name.lower() or "arducam" in name.lower()
    ]

    # If we can see a Reachy camera, do NOT silently fall back to the laptop webcam.
    # Only fall back when there is no Reachy-like camera detected at all.
    to_try = candidates if candidates else devices

    for dev, name in to_try:
        idx = dev_to_index(dev)
        if idx is None:
            continue
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue

        # Warmup reads: some cameras need a few frames.
        ok = False
        deadline = time.monotonic() + (warmup_s if ("reachy" in name.lower() or "arducam" in name.lower()) else 0.6)
        while time.monotonic() < deadline:
            ret, frame = cap.read()
            if ret and frame is not None and frame.size:
                ok = True
                break
            time.sleep(0.05)

        if ok:
            print(f"Using camera {dev} ({name})")
            return cap

        cap.release()

    if candidates:
        raise RuntimeError(
            "Detected a Reachy camera, but OpenCV could not read frames from it.\n"
            "Common causes: the device is busy (a script was Ctrl+Z stopped), permissions, or the daemon already grabbed it.\n"
            "Try:\n"
            "- Check who uses it: `fuser -v /dev/video4`\n"
            "- Force device: `REACHY_CAMERA_DEVICE=4 python3 demo_imshow_head_ears.py`\n"
            "- Increase warmup: `REACHY_CAMERA_WARMUP=6 python3 demo_imshow_head_ears.py`\n"
        )
    raise RuntimeError(
        "Could not find a working camera via V4L2. Try setting REACHY_CAMERA_DEVICE=0 (or another index)."
    )


def main() -> None:
    host = os.getenv("REACHY_MINI_HOST", "127.0.0.1")
    port = int(os.getenv("REACHY_MINI_PORT", "8000"))
    wait_s = float(os.getenv("REACHY_MINI_WAIT", "10"))
    duration_s_env = os.getenv("REACHY_DEMO_DURATION", "")
    duration_s = float(duration_s_env) if duration_s_env else None

    _wait_for_tcp(host, port, wait_s)

    # Important on Linux: avoid ReachyMini's in-process GStreamer pipeline when
    # using OpenCV's Qt windows; it can spam Qt/GLib thread warnings.
    with ReachyMini(
        host=host,
        port=port,
        connection_mode="localhost_only" if host in {"127.0.0.1", "localhost"} else "network",
        timeout=max(5.0, wait_s),
        media_backend="no_media",
    ) as mini:
        print("Connected. Opening Reachy V4L2 camera and imshow window...")

        cap = _open_reachy_camera()

        # Optional resolution hint (camera may ignore).
        width = int(os.getenv("REACHY_CAM_WIDTH", "1280"))
        height = int(os.getenv("REACHY_CAM_HEIGHT", "720"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        cv2.namedWindow("ReachyMini", cv2.WINDOW_NORMAL)

        start = time.monotonic()
        step = 0

        print("Streaming camera. Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        frame,
                        "No frame from Reachy V4L2 camera",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("ReachyMini", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                t = time.monotonic() - start
                if duration_s is not None and t > duration_s:
                    break

                # Timed motion sequence (head + ears).
                if step == 0 and t > 0.5:
                    mini.goto_target(head=create_head_pose(yaw=25), duration=1.0, method="minjerk")
                    step += 1
                elif step == 1 and t > 2.0:
                    mini.goto_target(head=create_head_pose(yaw=-25), duration=1.0, method="minjerk")
                    step += 1
                elif step == 2 and t > 3.5:
                    mini.goto_target(head=create_head_pose(yaw=0), duration=1.0, method="minjerk")
                    step += 1
                elif step == 3 and t > 5.0:
                    mini.goto_target(antennas=np.deg2rad([45, -45]), duration=0.6, method="minjerk")
                    step += 1
                elif step == 4 and t > 6.0:
                    mini.goto_target(antennas=np.deg2rad([-45, 45]), duration=0.6, method="minjerk")
                    step += 1
                elif step == 5 and t > 7.0:
                    mini.goto_target(antennas=np.deg2rad([0, 0]), duration=0.6, method="minjerk")
                    step += 1

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
