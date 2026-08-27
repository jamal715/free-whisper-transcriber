from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import imageio_ffmpeg
from mutagen import File as MutagenFile

# Groq free tier allows 25 MB attachments. Stay comfortably below it.
MAX_CHUNK_BYTES = 20 * 1024 * 1024
TARGET_CHUNK_BYTES = 16 * 1024 * 1024

SUPPORTED_OUTPUT_EXTENSIONS = {
    ".m4a": ".m4a",
    ".mp4": ".m4a",
    ".mp3": ".mp3",
    ".mpga": ".mp3",
    ".mpeg": ".mp3",
    ".ogg": ".ogg",
    ".webm": ".webm",
    ".wav": ".wav",
    ".flac": ".flac",
}


def _ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_duration(path: str) -> float:
    try:
        media = MutagenFile(path)
        info = getattr(media, "info", None)
        return max(0.0, float(getattr(info, "length", 0.0) or 0.0))
    except Exception:
        return 0.0


def _segment_once(input_path: str, output_dir: Path, segment_seconds: int, output_ext: str) -> list[Path]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(output_dir / f"chunk_%03d{output_ext}")
    command = [
        _ffmpeg(),
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-threads", "1",
        "-i", input_path,
        "-map", "0:a:0",
        "-vn",
        "-c:a", "copy",
        "-f", "segment",
        "-segment_time", str(segment_seconds),
        "-reset_timestamps", "1",
        pattern,
    ]

    proc = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip()
        raise RuntimeError(
            "Could not split this recording without re-encoding it. "
            "Please use M4A or MP3 for the most reliable large-file support."
            + (f" Details: {detail[-500:]}" if detail else "")
        )

    return sorted(output_dir.glob("chunk_*"))


def split_audio_lossless(input_path: str, output_dir: str) -> list[dict]:
    source = Path(input_path)
    size_bytes = source.stat().st_size
    duration = probe_duration(str(source))

    if duration <= 0:
        raise RuntimeError(
            "Could not read the recording duration. Please use M4A or MP3."
        )

    output_ext = SUPPORTED_OUTPUT_EXTENSIONS.get(source.suffix.lower(), ".m4a")

    # Estimate a chunk duration from the actual average source bitrate.
    # 16 MB target gives plenty of headroom below Groq's 25 MB attachment cap.
    seconds = int(duration * (TARGET_CHUNK_BYTES / max(1, size_bytes)))
    segment_seconds = max(60, min(900, seconds))

    out_dir = Path(output_dir)

    # If container overhead or bitrate variation creates an oversized piece,
    # retry with shorter segments. Stream-copying does not decode/re-encode audio.
    for _ in range(5):
        paths = _segment_once(
            input_path=str(source),
            output_dir=out_dir,
            segment_seconds=segment_seconds,
            output_ext=output_ext,
        )
        if paths and all(p.stat().st_size <= MAX_CHUNK_BYTES for p in paths):
            break
        segment_seconds = max(30, segment_seconds // 2)
    else:
        raise RuntimeError(
            "The recording could not be split below the transcription service's size limit."
        )

    if not paths:
        raise RuntimeError("No audio pieces were produced.")

    chunks = []
    offset = 0.0
    for path in paths:
        chunk_duration = probe_duration(str(path))
        if chunk_duration <= 0:
            chunk_duration = float(segment_seconds)

        chunks.append({
            "path": str(path),
            "duration": chunk_duration,
            "offset": offset,
        })
        offset += chunk_duration

    return chunks
