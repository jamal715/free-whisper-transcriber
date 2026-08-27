from __future__ import annotations

from pathlib import Path
import subprocess


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
    proc = _run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        timeout=60,
    )
    try:
        return max(0.0, float((proc.stdout or "0").strip()))
    except ValueError:
        return 0.0


def split_to_audio_chunks(
    input_path: str,
    output_dir: str,
    segment_seconds: int = 1200,
    bitrate: str = "32k",
) -> list[dict]:
    out_dir = Path(output_dir)
    pattern = str(out_dir / "chunk_%03d.mp3")

    _run(
        [
            "ffmpeg",
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
    for path in paths:
        duration = probe_duration(str(path))
        if duration <= 0:
            continue
        chunks.append({
            "path": str(path),
            "duration": duration,
            "offset": offset,
        })
        offset += duration

    return chunks
