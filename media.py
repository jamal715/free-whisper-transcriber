from __future__ import annotations

from pathlib import Path
import subprocess

import imageio_ffmpeg
from mutagen import File as MutagenFile


def _ffmpeg_exe() -> str:
    """Return a bundled FFmpeg executable supplied by imageio-ffmpeg."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(command: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        error = (proc.stderr or "").strip()
        raise RuntimeError(
            "FFmpeg could not process this recording."
            + (f" Details: {error[-700:]}" if error else "")
        )
    return proc


def probe_duration(path: str) -> float:
    """Read media duration without requiring a separate ffprobe binary."""
    try:
        media = MutagenFile(path)
        info = getattr(media, "info", None)
        length = getattr(info, "length", 0.0)
        return max(0.0, float(length or 0.0))
    except Exception:
        return 0.0


def split_to_audio_chunks(
    input_path: str,
    output_dir: str,
    segment_seconds: int = 1200,
    bitrate: str = "32k",
) -> list[dict]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "chunk_%03d.mp3")

    _run(
        [
            _ffmpeg_exe(),
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-threads", "1",
            "-i", input_path,
            "-map", "0:a:0",
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "libmp3lame",
            "-b:a", bitrate,
            "-f", "segment",
            "-segment_time", str(segment_seconds),
            "-reset_timestamps", "1",
            pattern,
        ]
    )

    paths = sorted(out_dir.glob("chunk_*.mp3"))
    chunks = []
    offset = 0.0

    for index, path in enumerate(paths):
        duration = probe_duration(str(path))
        if duration <= 0:
            # All but the last segment are fixed-length by construction.
            duration = float(segment_seconds)

        chunks.append(
            {
                "path": str(path),
                "duration": duration,
                "offset": offset,
            }
        )
        offset += duration

    if not chunks:
        raise RuntimeError("No audio chunks were produced from this recording.")

    return chunks
