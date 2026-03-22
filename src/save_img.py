from __future__ import annotations

import os
import socket
import time
from datetime import datetime
from pathlib import Path

import cv2
from reachy_mini import ReachyMini


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


def main() -> None:
    host = os.getenv("REACHY_MINI_HOST", "127.0.0.1")
    port = int(os.getenv("REACHY_MINI_PORT", "8000"))
    wait_s = float(os.getenv("REACHY_MINI_WAIT", "10"))

    captures_dir = Path(__file__).resolve().parent / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = captures_dir / f"reachy_{timestamp}.png"

    _wait_for_tcp(host, port, wait_s)

    with ReachyMini(
        host=host,
        port=port,
        connection_mode="localhost_only" if host in {"127.0.0.1", "localhost"} else "network",
        timeout=max(5.0, wait_s),
    ) as mini:
        print("Connecting to Reachy and waking up camera...")
        time.sleep(1.5)

        frame = mini.media.get_frame()

    if frame is None:
        raise RuntimeError(
            "Reachy failed to grab a frame. Check that the camera is connected "
            "and that the Reachy Mini media backend is running."
        )

    ok = cv2.imwrite(str(out_path), frame)
    if not ok:
        raise RuntimeError(f"Failed to write image to {out_path}")

    print(f"Saved image: {out_path}")


if __name__ == "__main__":
    main()
