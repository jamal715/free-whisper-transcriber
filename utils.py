from __future__ import annotations


def format_seconds(seconds: float) -> str:
    milliseconds = int(round(max(0.0, seconds) * 1000))
    total_seconds, ms = divmod(milliseconds, 1000)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _srt_timestamp(seconds: float) -> str:
    milliseconds = int(round(max(0.0, seconds) * 1000))
    total_seconds, ms = divmod(milliseconds, 1000)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def transcript_to_srt(segments: list[dict]) -> str:
    blocks = []
    for idx, seg in enumerate(segments, start=1):
        text = seg["text"].strip()
        if not text:
            continue
        blocks.append(
            f"{idx}\n"
            f"{_srt_timestamp(seg['start'])} --> {_srt_timestamp(seg['end'])}\n"
            f"{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")
