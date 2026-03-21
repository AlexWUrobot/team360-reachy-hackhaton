import os
import socket
import time
import wave
from pathlib import Path

import numpy as np

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


def _float_to_int16(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype(np.int16)


def _write_wav_mono_int16(path: Path, pcm16: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def main() -> None:
    host = os.getenv("REACHY_MINI_HOST", "127.0.0.1")
    port = int(os.getenv("REACHY_MINI_PORT", "8000"))
    wait_s = float(os.getenv("REACHY_MINI_WAIT", "10"))

    record_s = float(os.getenv("REACHY_RECORD_SECONDS", "5"))
    media_backend = os.getenv("REACHY_MEDIA_BACKEND", "gstreamer")

    out_path = Path(os.getenv("REACHY_RECORD_OUT", "/tmp/reachy_mic_5s.wav"))

    _wait_for_tcp(host, port, wait_s)

    with ReachyMini(
        host=host,
        port=port,
        connection_mode="localhost_only" if host in {"127.0.0.1", "localhost"} else "network",
        timeout=max(5.0, wait_s),
        media_backend=media_backend,
    ) as mini:
        sr = mini.media.get_input_audio_samplerate()
        if sr is None or sr <= 0:
            raise RuntimeError(
                f"Could not determine input audio samplerate (got {sr}). "
                "Audio may be disabled or the backend may be audio-only without input."
            )

        print(f"Recording from Reachy mic for {record_s:.1f}s at {sr} Hz...")
        mini.media.start_recording()

        chunks: list[np.ndarray] = []
        deadline = time.monotonic() + record_s

        try:
            while time.monotonic() < deadline:
                chunk = mini.media.get_audio_sample()
                if chunk is not None and len(chunk) > 0:
                    arr = np.asarray(chunk, dtype=np.float32)
                    # The backend returns stereo chunks shaped (N, 2) on many setups.
                    # Downmix to mono so the WAV duration/pitch is correct.
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        arr = arr.mean(axis=1)
                    elif arr.ndim == 2 and arr.shape[1] == 1:
                        arr = arr[:, 0]
                    else:
                        arr = arr.reshape(-1)
                    chunks.append(arr)
                else:
                    time.sleep(0.01)
        finally:
            # Not exposed on MediaManager, but available on the backend.
            if getattr(mini.media, "audio", None) is not None:
                mini.media.audio.stop_recording()  # type: ignore[attr-defined]

        if not chunks:
            raise RuntimeError(
                "No audio samples received. Check that Reachy Mini microphone is available and not muted."
            )

        audio = np.concatenate(chunks)
        pcm16 = _float_to_int16(audio)

        rms = float(np.sqrt(np.mean((pcm16.astype(np.float32) / 32768.0) ** 2)))
        peak = float(np.max(np.abs(pcm16)) / 32768.0)
        seconds = float(len(pcm16) / sr)
        print(f"Recorded stats: seconds={seconds:.2f}, rms={rms:.4f}, peak={peak:.4f}")

        _write_wav_mono_int16(out_path, pcm16, sr)
        print(f"Saved WAV: {out_path} ({seconds:.2f}s)")

        print("Playing back on Reachy speaker...")
        mini.media.play_sound(str(out_path))

        # Important: keep the program alive while audio plays.
        # Exiting the context manager closes MediaManager and stops playback.
        print(f"Waiting {seconds:.2f}s for playback...")
        time.sleep(seconds + 0.5)
        print("Done.")


if __name__ == "__main__":
    main()
