import os
import socket
import time

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

    _wait_for_tcp(host, port, wait_s)

    with ReachyMini(
        host=host,
        port=port,
        connection_mode="localhost_only" if host in {"127.0.0.1", "localhost"} else "network",
        timeout=max(5.0, wait_s),
    ) as mini:
        print("Connected to Reachy Mini!")

        print("Wiggling antennas...")
        mini.goto_target(antennas=[0.5, -0.5], duration=0.5)
        mini.goto_target(antennas=[-0.5, 0.5], duration=0.5)
        mini.goto_target(antennas=[0, 0], duration=0.5)

        print("Done!")


if __name__ == "__main__":
    main()