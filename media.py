from __future__ import annotations

from math import ceil
from pathlib import Path
import shutil
import subprocess

import imageio_ffmpeg
from mutagen import File as MutagenFile

# Groq free-tier direct attachments are capped at 25 MB. Stay well below
# that so multipart overhead and container variance never push a chunk over.
MAX_CHUNK_BYTES = 19 * 1024 * 1024
TARGET_CHUNK_BYTES = 16 * 1024 * 1024
OVERLAP_SECONDS = 2.0

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


def _extract_window(
    *,
    input_path: str,
    output_path: str,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    command = [
        _ffmpeg(),
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-threads", "1",
        "-ss", f"{max(0.0, start_seconds):.3f}",
        "-i", input_path,
        "-t", f"{max(0.1, duration_seconds):.3f}",
        "-map", "0:a:0",
        "-vn",
        "-c:a", "copy",
        output_path,
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
            "M4A and MP3 are the most reliable formats for very large interviews."
            + (f" Details: {detail[-500:]}" if detail else "")
        )


def _build_chunks(
    *,
    input_path: str,
    output_dir: Path,
    source_duration: float,
    segment_seconds: int,
    output_ext: str,
) -> list[dict]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = max(1, ceil(source_duration / segment_seconds))
    chunks: list[dict] = []

    for i in range(count):
        nominal_start = float(i * segment_seconds)
        nominal_end = min(source_duration, float((i + 1) * segment_seconds))
        chunk_start = max(0.0, nominal_start - (OVERLAP_SECONDS if i else 0.0))
        chunk_end = nominal_end

        path = output_dir / f"chunk_{i:03d}{output_ext}"
        _extract_window(
            input_path=input_path,
            output_path=str(path),
            start_seconds=chunk_start,
            duration_seconds=max(0.1, chunk_end - chunk_start),
        )

        chunk_duration = probe_duration(str(path))
        if chunk_duration <= 0:
            chunk_duration = max(0.1, chunk_end - chunk_start)

        chunks.append(
            {
                "path": str(path),
                "duration": chunk_duration,
                "offset": chunk_start,
                "keep_after": nominal_start if i else 0.0,
                "nominal_end": nominal_end,
            }
        )

    return chunks


def split_audio_lossless(input_path: str, output_dir: str) -> list[dict]:
    source = Path(input_path)
    size_bytes = source.stat().st_size
    duration = probe_duration(str(source))

    if duration <= 0:
        raise RuntimeError(
            "Could not read the recording duration. Please use M4A or MP3 for large files."
        )

    output_ext = SUPPORTED_OUTPUT_EXTENSIONS.get(source.suffix.lower(), ".m4a")

    # Estimate chunk length from the source's actual average bitrate.
    seconds = int(duration * (TARGET_CHUNK_BYTES / max(1, size_bytes)))
    segment_seconds = max(90, min(900, seconds))
    out_dir = Path(output_dir)

    # Retry with shorter windows if a variable-bitrate section is unexpectedly large.
    for _ in range(6):
        chunks = _build_chunks(
            input_path=str(source),
            output_dir=out_dir,
            source_duration=duration,
            segment_seconds=segment_seconds,
            output_ext=output_ext,
        )
        if chunks and all(Path(c["path"]).stat().st_size <= MAX_CHUNK_BYTES for c in chunks):
            return chunks
        segment_seconds = max(45, int(segment_seconds * 0.72))

    raise RuntimeError(
        "The recording could not be split below the transcription service's file-size limit. "
        "Convert it to M4A or MP3 and retry."
    )
