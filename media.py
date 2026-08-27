from __future__ import annotations

import subprocess


def prepare_audio(input_path: str, output_path: str) -> None:
    """
    Convert any supported media file into a lightweight Whisper-ready stream:
    mono, 16 kHz, signed 16-bit PCM. FFmpeg is limited to one thread.
    """
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-threads", "1",
        "-i", input_path,
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        output_path,
    ]
    proc = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        error = (proc.stderr or "").strip()
        raise RuntimeError(
            "Could not extract audio from the uploaded file."
            + (f" FFmpeg: {error[-600:]}" if error else "")
        )


def probe_duration(path: str) -> float:
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=30,
    )
    try:
        return max(0.0, float((proc.stdout or "0").strip()))
    except ValueError:
        return 0.0
