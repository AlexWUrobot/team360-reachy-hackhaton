import os
import socket
import time

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


def _do_and_wait(action_name: str, fn, duration_s: float) -> None:
    print(action_name)
    fn()
    # Give the robot time to finish the motion before the next command.
    time.sleep(max(0.0, duration_s) + 0.2)


def main() -> None:
    host = os.getenv("REACHY_MINI_HOST", "127.0.0.1")
    port = int(os.getenv("REACHY_MINI_PORT", "8000"))
    wait_s = float(os.getenv("REACHY_MINI_WAIT", "10"))

    # Movement parameters (override with env vars if needed)
    head_yaw_deg = float(os.getenv("HEAD_YAW_DEG", "90"))
    antenna_deg = float(os.getenv("ANTENNA_DEG", "45"))
    body_z_mm = float(os.getenv("BODY_Z_MM", "20"))  # back around 10 mm  max 20

    # Durations
    head_duration = float(os.getenv("HEAD_DURATION", "1.5"))
    antenna_duration = float(os.getenv("ANTENNA_DURATION", "1.0"))
    body_duration = float(os.getenv("BODY_DURATION", "2.0"))

    _wait_for_tcp(host, port, wait_s)

    with ReachyMini(
        host=host,
        port=port,
        connection_mode="localhost_only" if host in {"127.0.0.1", "localhost"} else "network",
        timeout=max(5.0, wait_s),
        media_backend="no_media",
    ) as mini:
        print("Connected. Starting serial movement sequence...")

        # 1) Head: +30 then -30 (yaw in degrees)
        _do_and_wait(
            f"Head yaw +{head_yaw_deg:.0f}°",
            lambda: mini.goto_target(
                head=create_head_pose(yaw=head_yaw_deg, degrees=True),
                duration=head_duration,
                method="minjerk",
            ),
            head_duration,
        )
        _do_and_wait(
            f"Head yaw -{head_yaw_deg:.0f}°",
            lambda: mini.goto_target(
                head=create_head_pose(yaw=-head_yaw_deg, degrees=True),
                duration=head_duration,
                method="minjerk",
            ),
            head_duration,
        )
        _do_and_wait(
            "Head yaw back to 0°",
            lambda: mini.goto_target(
                head=create_head_pose(yaw=0, degrees=True),
                duration=head_duration,
                method="minjerk",
            ),
            head_duration,
        )

        # 2) Antennas: +45 then -45 (in degrees -> radians)
        _do_and_wait(
            f"Antennas to +{antenna_deg:.0f}° / -{antenna_deg:.0f}°",
            lambda: mini.goto_target(
                antennas=np.deg2rad([antenna_deg, -antenna_deg]),
                duration=antenna_duration,
                method="minjerk",
            ),
            antenna_duration,
        )
        _do_and_wait(
            f"Antennas to -{antenna_deg:.0f}° / +{antenna_deg:.0f}°",
            lambda: mini.goto_target(
                antennas=np.deg2rad([-antenna_deg, antenna_deg]),
                duration=antenna_duration,
                method="minjerk",
            ),
            antenna_duration,
        )
        _do_and_wait(
            "Antennas back to 0° / 0°",
            lambda: mini.goto_target(
                antennas=np.deg2rad([0, 0]),
                duration=antenna_duration,
                method="minjerk",
            ),
            antenna_duration,
        )

        # 3) Body up/down: on Reachy Mini this is typically the stewart platform,
        # controlled via the head pose translation (z in mm).
        _do_and_wait(
            f"Body up (+{body_z_mm:.0f}mm)",
            lambda: mini.goto_target(
                head=create_head_pose(z=body_z_mm, mm=True),
                duration=body_duration,
                method="minjerk",
            ),
            body_duration,
        )
        _do_and_wait(
            f"Body down (-{body_z_mm:.0f}mm)",
            lambda: mini.goto_target(
                head=create_head_pose(z=-body_z_mm, mm=True),
                duration=body_duration,
                method="minjerk",
            ),
            body_duration,
        )
        _do_and_wait(
            "Body back to neutral (z=0mm)",
            lambda: mini.goto_target(
                head=create_head_pose(z=0, mm=True),
                duration=body_duration,
                method="minjerk",
            ),
            body_duration,
        )

        print("Done.")


if __name__ == "__main__":
    main()
