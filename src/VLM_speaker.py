import argparse
import os
import socket
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor

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


def _save_frame_with_timestamp(frame_bgr: np.ndarray) -> Path:
    out_dir = Path(os.getenv("REACHY_CAPTURE_DIR", "").strip() or (Path(__file__).resolve().parent / "captures"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
    out_path = out_dir / f"reachy_{ts}.jpg"

    ok = cv2.imwrite(str(out_path), frame_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image to: {out_path}")
    return out_path


def _has_accelerate() -> bool:
    try:
        import accelerate  # noqa: F401

        return True
    except Exception:
        return False


def _load_vlm(model_id: str, device: str, use_device_map: bool):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    device_map = None
    if device.startswith("cuda") and use_device_map and _has_accelerate():
        device_map = "auto"

    load_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }

    model = None

    # Dedicated Qwen3-VL class when available.
    if getattr(cfg, "model_type", "") in {"qwen3_vl", "qwen3-vl"}:
        try:
            from transformers import Qwen3VLForConditionalGeneration  # type: ignore

            model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        except Exception:
            model = None

    if model is None:
        model = AutoModel.from_pretrained(model_id, **load_kwargs)

    if device_map is None:
        model = model.to(device)

    model.eval()
    return processor, model


def _prepare_inputs(processor, image: Image.Image, prompt: str):
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


def _tts_to_mp3_gtts(text: str, out_mp3: Path, lang: str) -> None:
    try:
        from gtts import gTTS  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: gTTS. Install with: pip install gTTS\n"
            "Note: gTTS uses an online service, so internet is required.\n"
            f"Original error: {exc}"
        ) from exc

    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    gTTS(text=text, lang=lang).save(str(out_mp3))


def _tts_to_wav_kokoro(text: str, out_wav: Path, voice: str, speed: float) -> float:
    """Generate WAV using Kokoro, following its own CLI format (24kHz mono int16)."""
    try:
        import wave

        import numpy as np
        from kokoro import KPipeline  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing Kokoro dependencies. Install with: pip install kokoro\n"
            "Kokoro will also download model/voice files from HuggingFace on first run.\n"
            f"Original error: {exc}"
        ) from exc

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    lang_code = os.getenv("KOKORO_LANG", "").strip().lower() or voice[:1].lower()
    device = os.getenv("KOKORO_DEVICE", "").strip() or None

    pipeline = KPipeline(lang_code=lang_code, device=device)

    sr = 24000
    total_frames = 0

    with wave.open(str(out_wav), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)

        for result in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
            audio = getattr(result, "audio", None)
            if audio is None:
                output = getattr(result, "output", None)
                audio = getattr(output, "audio", None) if output is not None else None
            if audio is None:
                continue

            audio_np = audio.detach().cpu().numpy().astype(np.float32)
            pcm16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)
            wav_file.writeframes(pcm16.tobytes())
            total_frames += int(pcm16.shape[0])

    return total_frames / sr if sr > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM -> TTS -> Reachy speaker")
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
        "--tts-backend",
        default=os.getenv("TTS_BACKEND", "gtts"),
        choices=["gtts", "kokoro"],
        help="TTS backend: gtts (online mp3) or kokoro (offline wav)",
    )
    args = parser.parse_args()

    # Confirm daemon is running (needed for speaker playback).
    host = os.getenv("REACHY_MINI_HOST", "127.0.0.1")
    port = int(os.getenv("REACHY_MINI_PORT", "8000"))
    _wait_for_tcp(host, port, float(os.getenv("REACHY_MINI_WAIT", "5")))

    # 1) Capture image
    cap = _open_reachy_camera()
    try:
        frame_bgr = _grab_one_frame_bgr(cap)
    finally:
        cap.release()

    saved_path = _save_frame_with_timestamp(frame_bgr)
    print(f"Saved image: {saved_path}")

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    # 2) Run VLM
    print(f"Loading model: {args.model}")
    processor, model = _load_vlm(args.model, args.device, use_device_map=(not args.no_device_map))

    inputs = _prepare_inputs(processor, pil, args.prompt)

    if hasattr(model, "device"):
        device = model.device
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    decoded = processor.batch_decode(out, skip_special_tokens=True)
    answer = "\n".join([t.strip() for t in decoded if t.strip()])
    if not answer:
        raise SystemExit("VLM returned empty output")

    print("VLM output:")
    print(answer)

    # 3) TTS
    out_file: Path
    duration_s: float | None = None

    if args.tts_backend == "kokoro":
        voice = os.getenv("KOKORO_VOICE", "af_heart")
        speed = float(os.getenv("KOKORO_SPEED", "1.0"))
        out_file = Path(os.getenv("TTS_OUT", "/tmp/reachy_vlm_tts.wav"))
        print(f"Generating Kokoro WAV -> {out_file}")
        duration_s = _tts_to_wav_kokoro(answer, out_file, voice=voice, speed=speed)
    else:
        lang = os.getenv("TTS_LANG", "en")
        out_file = Path(os.getenv("TTS_OUT", "/tmp/reachy_vlm_tts.mp3"))
        print(f"Generating gTTS MP3 -> {out_file}")
        _tts_to_mp3_gtts(answer, out_file, lang)
        # heuristic (gtts does not give duration)
        duration_s = max(2.0, min(30.0, 0.07 * len(answer)))

    play_wait_s = float(os.getenv("TTS_PLAY_WAIT", str((duration_s or 3.0) + 0.5)))

    # 4) Play on Reachy
    media_backend = os.getenv("REACHY_MEDIA_BACKEND", "gstreamer_no_video")
    with ReachyMini(
        host=host,
        port=port,
        connection_mode="localhost_only" if host in {"127.0.0.1", "localhost"} else "network",
        timeout=max(5.0, float(os.getenv("REACHY_MINI_WAIT", "5"))),
        media_backend=media_backend,
    ) as mini:
        mini.media.play_sound(str(out_file))
        print(f"Speaking... waiting {play_wait_s:.2f}s")
        time.sleep(play_wait_s)


if __name__ == "__main__":
    main()
