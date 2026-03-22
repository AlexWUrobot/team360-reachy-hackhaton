import os
import signal
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
        video = name_path.parent.name  # video6
        dev = f"/dev/{video}"
        try:
            name = name_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        devices.append((dev, name))
    return devices


def _open_reachy_camera() -> cv2.VideoCapture:
    def dev_to_index(dev: str) -> int | None:
        base = Path(dev).name
        if base.startswith("video") and base[5:].isdigit():
            return int(base[5:])
        return None

    forced = os.getenv("REACHY_CAMERA_DEVICE", "").strip()
    if forced:
        idx = int(forced) if forced.isdigit() else dev_to_index(forced)
        if idx is None:
            raise RuntimeError(
                "REACHY_CAMERA_DEVICE must be an integer (e.g. '6') or /dev/videoX. "
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

    to_try = candidates if candidates else devices

    for dev, name in to_try:
        idx = dev_to_index(dev)
        if idx is None:
            continue
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue

        ok = False
        deadline = time.monotonic() + (warmup_s if ("reachy" in name.lower()) else 0.6)
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

    raise RuntimeError(
        "Could not find a working Reachy camera via V4L2. "
        "Try setting REACHY_CAMERA_DEVICE=6 (or the current /dev/videoX index)."
    )


def _load_yolo():
    # Ultralytics YOLO is the easiest way to detect both person and cat (COCO classes).
    # Install once in your venv:
    #   pip install ultralytics
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: ultralytics. Install with: pip install ultralytics\n"
            f"Original error: {exc}"
        ) from exc

    model_path = os.getenv("YOLO_MODEL", "yolov8n.pt")
    return YOLO(model_path)


def main() -> None:
    host = os.getenv("REACHY_MINI_HOST", "127.0.0.1")
    port = int(os.getenv("REACHY_MINI_PORT", "8000"))
    wait_s = float(os.getenv("REACHY_MINI_WAIT", "10"))

    # detection config
    conf = float(os.getenv("YOLO_CONF", "0.35"))
    imgsz = int(os.getenv("YOLO_IMGSZ", "640"))
    classes_str = os.getenv("YOLO_CLASSES", "cat,person")
    want = {c.strip().lower() for c in classes_str.split(",") if c.strip()}
    # Runtime-selectable target class:
    # - both: detect+track the best among want
    # - cat/person: only consider that class
    # Default to tracking people.
    target_mode = os.getenv("TARGET_MODE", "person").strip().lower()  # both|cat|person
    if target_mode not in {"both", "cat", "person"}:
        target_mode = "both"

    # tracking config
    tracking = os.getenv("REACHY_TRACKING", "1") not in {"0", "false", "False"}
    max_yaw_deg = float(os.getenv("REACHY_MAX_YAW", "90"))
    yaw_gain = float(os.getenv("REACHY_YAW_GAIN", "25"))  # degrees at full left/right
    update_hz = float(os.getenv("REACHY_UPDATE_HZ", "2"))
    deadband_deg = float(os.getenv("REACHY_YAW_DEADBAND", "2"))

    # Optional: also turn the body while tracking (so the robot "moves" with the person).
    body_follow = os.getenv("REACHY_BODY_FOLLOW", "1") not in {"0", "false", "False"}
    max_body_yaw_deg = float(os.getenv("REACHY_MAX_BODY_YAW", "45"))
    body_update_hz = float(os.getenv("REACHY_BODY_UPDATE_HZ", "1"))
    body_deadband_deg = float(os.getenv("REACHY_BODY_DEADBAND", "3"))
    body_step_deg = float(os.getenv("REACHY_BODY_STEP_DEG", "8"))

    _wait_for_tcp(host, port, wait_s)

    yolo = _load_yolo()

    stop = False

    def _handle_sigint(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sigint)
    # Ctrl+Z sends SIGTSTP on Linux; if the process is stopped it will keep the
    # V4L2 device busy and the next run won't be able to open the Reachy camera.
    # We treat it like a clean quit.
    if hasattr(signal, "SIGTSTP"):
        signal.signal(signal.SIGTSTP, _handle_sigint)

    with ReachyMini(
        host=host,
        port=port,
        connection_mode="localhost_only" if host in {"127.0.0.1", "localhost"} else "network",
        timeout=max(5.0, wait_s),
        media_backend="no_media",
    ) as mini:
        cap = _open_reachy_camera()

        cv2.namedWindow("Reachy - cat/person tracking", cv2.WINDOW_NORMAL)

        last_cmd_t = 0.0
        last_yaw = 0.0
        last_body_cmd_t = 0.0
        body_yaw_deg = 0.0

        print(
            "Running YOLO detection. Keys: q=quit, t=toggle tracking, 1=cat, 2=person, b=both"
        )

        try:
            while not stop:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # YOLO inference
                results = yolo.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
                r0 = results[0]

                best = None  # (conf, name, xyxy)
                names = getattr(r0, "names", {})

                if getattr(r0, "boxes", None) is not None and len(r0.boxes) > 0:
                    for b in r0.boxes:
                        cls_id = int(b.cls.item())
                        name = str(names.get(cls_id, cls_id)).lower()
                        score = float(b.conf.item())
                        if want and name not in want:
                            continue
                        if target_mode != "both" and name != target_mode:
                            continue
                        xyxy = b.xyxy[0].cpu().numpy().astype(int)
                        cand = (score, name, xyxy)
                        if best is None or cand[0] > best[0]:
                            best = cand

                h, w = frame.shape[:2]
                if best is not None:
                    score, name, (x1, y1, x2, y2) = best
                    x1 = int(np.clip(x1, 0, w - 1))
                    x2 = int(np.clip(x2, 0, w - 1))
                    y1 = int(np.clip(y1, 0, h - 1))
                    y2 = int(np.clip(y2, 0, h - 1))

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    offset_x = (cx - (w / 2.0)) / (w / 2.0)  # -1 .. +1

                    # draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f"{name} {score:.2f}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    # Tracking: map horizontal offset to yaw.
                    # We split the required yaw between body_yaw and head yaw:
                    # - body_yaw follows slowly and keeps the head within a comfortable range
                    # - head yaw reacts faster
                    target_total_yaw = float(np.clip(-offset_x * yaw_gain, -max_body_yaw_deg, max_body_yaw_deg))

                    # Head yaw tries to correct the remaining yaw after body_yaw.
                    desired_yaw = float(
                        np.clip(target_total_yaw - body_yaw_deg, -max_yaw_deg, max_yaw_deg)
                    )
                    now = time.monotonic()

                    if tracking and body_follow and (now - last_body_cmd_t) >= (1.0 / max(body_update_hz, 0.1)):
                        # If the head is near its limit, start turning the body in small steps.
                        if abs(target_total_yaw - body_yaw_deg) > (max_yaw_deg * 0.8):
                            direction = 1.0 if (target_total_yaw - body_yaw_deg) > 0 else -1.0
                            next_body = float(
                                np.clip(
                                    body_yaw_deg + direction * body_step_deg,
                                    -max_body_yaw_deg,
                                    max_body_yaw_deg,
                                )
                            )
                        else:
                            next_body = float(np.clip(body_yaw_deg, -max_body_yaw_deg, max_body_yaw_deg))

                        if abs(next_body - body_yaw_deg) >= body_deadband_deg:
                            mini.goto_target(
                                body_yaw=np.deg2rad(next_body),
                                duration=0.9,
                                method="minjerk",
                            )
                            body_yaw_deg = next_body
                            last_body_cmd_t = now

                    if tracking and (now - last_cmd_t) >= (1.0 / max(update_hz, 0.1)):
                        if abs(desired_yaw - last_yaw) >= deadband_deg:
                            mini.goto_target(
                                head=create_head_pose(yaw=desired_yaw, degrees=True),
                                duration=0.6,
                                method="minjerk",
                            )
                            last_cmd_t = now
                            last_yaw = desired_yaw

                    cv2.putText(
                        frame,
                        f"mode={target_mode} head_yaw={last_yaw:.1f} body_yaw={body_yaw_deg:.1f} tracking={'on' if tracking else 'off'}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        frame,
                        f"No target (mode={target_mode}, want={', '.join(sorted(want))})  tracking={'on' if tracking else 'off'}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("Reachy - cat/person tracking", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("t"):
                    tracking = not tracking
                if key == ord("1"):
                    target_mode = "cat"
                if key == ord("2"):
                    target_mode = "person"
                if key == ord("b"):
                    target_mode = "both"

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
