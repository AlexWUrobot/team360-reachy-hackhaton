import argparse
import os
import socket
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor


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
        video = name_path.parent.name
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
        "Try setting REACHY_CAMERA_DEVICE to the current /dev/videoX index."
    )


def _grab_one_frame_bgr(cap: cv2.VideoCapture, warmup_frames: int = 5) -> np.ndarray:
    for _ in range(max(0, warmup_frames)):
        cap.read()
        time.sleep(0.02)

    for _ in range(50):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size:
            return frame
        time.sleep(0.05)

    raise RuntimeError("Failed to grab a frame from Reachy camera")


def _has_accelerate() -> bool:
    try:
        import accelerate  # noqa: F401

        return True
    except Exception:
        return False


def _load_model(model_id: str, device: str, use_device_map: bool):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    model = None
    device_map = None
    if device.startswith("cuda") and use_device_map and _has_accelerate():
        device_map = "auto"

    load_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        # Keep torch_dtype for maximum compatibility across transformers versions.
        "torch_dtype": dtype,
    }

    # Try the dedicated class if it exists in this transformers version.
    # Qwen3-VL models require a vision-language conditional generation model, not AutoModelForCausalLM.
    if getattr(cfg, "model_type", "") in {"qwen3_vl", "qwen3-vl"}:
        try:
            from transformers import Qwen3VLForConditionalGeneration  # type: ignore

            model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        except Exception:
            model = None

    # Generic conditional-generation fallback.
    if model is None:
        try:
            from transformers import AutoModelForConditionalGeneration  # type: ignore

            model = AutoModelForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        except Exception:
            model = None

    # Some transformers versions expose AutoModelForVision2Seq.
    if model is None:
        try:
            import transformers

            vision2seq = getattr(transformers, "AutoModelForVision2Seq", None)
            if vision2seq is not None:
                model = vision2seq.from_pretrained(model_id, **load_kwargs)
        except Exception:
            model = None

    # Last resort: base AutoModel (may not support .generate).
    if model is None:
        model = AutoModel.from_pretrained(model_id, **load_kwargs)

    # If we didn't use device_map sharding, explicitly move model to the requested device.
    if device_map is None:
        model = model.to(device)

    model.eval()
    return processor, model


def _prepare_inputs(processor, image: Image.Image, prompt: str):
    # Prefer chat template when available.
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        return processor(text=[text], images=[image], return_tensors="pt")

    return processor(text=[prompt], images=[image], return_tensors="pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen VL on Reachy Mini camera")
    parser.add_argument(
        "--model",
        default=os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-2B-Instruct"),
        help="HF model id (default: Qwen/Qwen3-VL-2B-Instruct)",
    )
    parser.add_argument(
        "--prompt",
        default=os.getenv("VLM_PROMPT", "Describe the image briefly."),
        help="Prompt/question to ask the VLM",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.getenv("VLM_MAX_NEW_TOKENS", "64")),
    )
    parser.add_argument(
        "--device",
        default=os.getenv("VLM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        help="cpu or cuda",
    )
    parser.add_argument(
        "--no-device-map",
        action="store_true",
        help="Disable accelerate device_map loading (use plain .to(device))",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the captured frame in an OpenCV window",
    )
    args = parser.parse_args()

    # Ensure daemon is up (also useful to confirm robot is connected). Not strictly required for camera capture.
    host = os.getenv("REACHY_MINI_HOST", "127.0.0.1")
    port = int(os.getenv("REACHY_MINI_PORT", "8000"))
    _wait_for_tcp(host, port, float(os.getenv("REACHY_MINI_WAIT", "5")))

    cap = _open_reachy_camera()
    try:
        frame_bgr = _grab_one_frame_bgr(cap)
    finally:
        cap.release()

    if args.show:
        cv2.imshow("Reachy frame", frame_bgr)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    print(f"Loading model: {args.model}")
    try:
        processor, model = _load_model(
            args.model, args.device, use_device_map=(not args.no_device_map)
        )
    except Exception as exc:
        msg = str(exc)
        if "requires `accelerate`" in msg or "pip install accelerate" in msg:
            raise SystemExit(
                "Model loading tried to use `device_map`, which requires `accelerate`.\n\n"
                "Fix options:\n"
                "- Install accelerate: pip install accelerate\n"
                "- Or run with: python3 VLM.py --no-device-map\n"
                "- Or run on CPU: python3 VLM.py --device cpu\n\n"
                f"Original error: {exc}"
            )
        if "Repository Not Found" in msg or "is not a valid model identifier" in msg or "404" in msg:
            raise SystemExit(
                "Failed to load the Hugging Face model repo.\n\n"
                "The repo id you used looks invalid or not accessible.\n"
                "Try one of these known-good public model ids:\n"
                "- Qwen/Qwen3-VL-2B-Instruct\n"
                "- Qwen/Qwen2-VL-2B-Instruct\n\n"
                "If you are trying to use a gated/private repo, run `huggingface-cli login` (or `hf auth login`).\n"
                f"Original error: {exc}"
            )
        raise

    inputs = _prepare_inputs(processor, pil, args.prompt)

    # Move tensors to the appropriate device if model isn't sharded.
    if hasattr(model, "device"):
        device = model.device
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    text = processor.batch_decode(out, skip_special_tokens=True)
    print("\n".join(text))


if __name__ == "__main__":
    main()
