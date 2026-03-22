import os
import socket
import sys
import time
from pathlib import Path

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


def _tts_to_mp3(text: str, out_mp3: Path, lang: str) -> None:
    try:
        from gtts import gTTS  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: gTTS. Install in your active env with:\n"
            "  pip install gTTS\n\n"
            "Note: gTTS uses an online service, so internet is required.\n"
            f"Original error: {exc}"
        ) from exc

    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    tts = gTTS(text=text, lang=lang)
    tts.save(str(out_mp3))


def _tts_to_wav_kokoro(text: str, out_wav: Path, voice: str, speed: float) -> float:
    """Generate WAV using the Kokoro TTS model.

    Kokoro writes 24kHz mono WAV in its own CLI; we follow that.
    Returns duration seconds.
    """
    try:
        import wave

        import numpy as np
        from kokoro import KPipeline  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing Kokoro dependencies. Install with something like:\n"
            "  pip install kokoro\n\n"
            "Kokoro will also download model/voice files from HuggingFace on first run.\n"
            f"Original error: {exc}"
        ) from exc

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # Lang defaults to first letter of voice (e.g. 'af_heart' -> 'a')
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
    host = os.getenv("REACHY_MINI_HOST", "127.0.0.1")
    port = int(os.getenv("REACHY_MINI_PORT", "8000"))
    wait_s = float(os.getenv("REACHY_MINI_WAIT", "10"))

    # Audio-only backend (avoid camera init)
    media_backend = os.getenv("REACHY_MEDIA_BACKEND", "gstreamer_no_video")

    default_text = "Jimmy is from Purdue, and he is handsome and talented."

    # Input text: CLI args or interactive prompt (empty => default_text)
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        typed = input(
            f'Enter text to speak (press Enter for default: "{default_text}"): '
        ).strip()
        text = typed or default_text

    backend = os.getenv("TTS_BACKEND", "gtts").strip().lower()  # gtts|kokoro

    duration_s: float | None = None

    if backend == "kokoro":
        voice = os.getenv("KOKORO_VOICE", "af_heart")
        speed = float(os.getenv("KOKORO_SPEED", "1.0"))
        out_file = Path(os.getenv("TTS_OUT", "/tmp/reachy_tts.wav"))
        print(f"Generating Kokoro TTS WAV (voice={voice}, speed={speed}) -> {out_file} ...")
        duration_s = _tts_to_wav_kokoro(text, out_file, voice=voice, speed=speed)
        print("TTS ready.")
    else:
        lang = os.getenv("TTS_LANG", "en")
        out_file = Path(os.getenv("TTS_OUT", "/tmp/reachy_tts.mp3"))
        print(f"Generating gTTS MP3 (lang={lang}) -> {out_file} ...")
        _tts_to_mp3(text, out_file, lang)
        print("TTS ready.")

    # Keep script alive during playback (ReachyMini closes audio on exit).
    if duration_s is None:
        duration_s = max(2.0, min(30.0, 0.07 * len(text)))
    play_wait_s = float(os.getenv("TTS_PLAY_WAIT", str(duration_s + 0.5)))

    _wait_for_tcp(host, port, wait_s)

    with ReachyMini(
        host=host,
        port=port,
        connection_mode="localhost_only" if host in {"127.0.0.1", "localhost"} else "network",
        timeout=max(5.0, wait_s),
        media_backend=media_backend,
    ) as mini:
        print(f"Speaking: {text!r}")
        mini.media.play_sound(str(out_file))
        print(f"Waiting {play_wait_s:.2f}s for playback...")
        time.sleep(play_wait_s)


if __name__ == "__main__":
    main()
