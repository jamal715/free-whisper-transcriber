from __future__ import annotations

import mimetypes
import time
from pathlib import Path
from typing import Callable, Optional

import requests

MODEL = "whisper-large-v3-turbo"
BASE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def _transcribe_one(path: str, api_key: str, language: Optional[str]) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": MODEL,
        "response_format": "verbose_json",
        "temperature": "0",
    }
    if language:
        data["language"] = language

    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"

    for attempt in range(4):
        try:
            with open(path, "rb") as audio:
                response = requests.post(
                    BASE_URL,
                    headers=headers,
                    data=data,
                    files={"file": (Path(path).name, audio, mime)},
                    timeout=(30, 600),
                )
        except requests.RequestException as exc:
            if attempt == 3:
                raise RuntimeError(f"Network error while contacting Groq: {exc}") from exc
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 429 and attempt < 3:
            retry_after = response.headers.get("retry-after")
            try:
                wait = float(retry_after) if retry_after else 2 ** attempt
            except ValueError:
                wait = 2 ** attempt
            time.sleep(max(1.0, min(30.0, wait)))
            continue

        if not response.ok:
            try:
                detail = response.json()
            except Exception:
                detail = response.text[-1000:]
            raise RuntimeError(f"Groq returned HTTP {response.status_code}: {detail}")

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("Groq returned an unreadable response.") from exc

    raise RuntimeError("Groq could not complete the transcription request.")


def transcribe_chunks(
    chunks: list[dict],
    api_key: str,
    language: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    if not api_key:
        raise ValueError("A Groq API key is required.")

    all_segments = []
    text_parts = []
    total = len(chunks)

    for i, chunk in enumerate(chunks, start=1):
        if progress_callback:
            progress_callback(i - 1, total, f"Transcribing part {i} of {total}…")

        payload = _transcribe_one(
            path=chunk["path"],
            api_key=api_key,
            language=language,
        )

        text = str(payload.get("text", "") or "").strip()
        if text:
            text_parts.append(text)

        offset = float(chunk.get("offset", 0.0))
        raw_segments = payload.get("segments") or []

        if raw_segments:
            for seg in raw_segments:
                seg_text = str(seg.get("text", "") or "").strip()
                if not seg_text:
                    continue
                all_segments.append({
                    "start": offset + float(seg.get("start", 0.0) or 0.0),
                    "end": offset + float(seg.get("end", 0.0) or 0.0),
                    "text": seg_text,
                })
        elif text:
            all_segments.append({
                "start": offset,
                "end": offset + float(chunk.get("duration", 0.0)),
                "text": text,
            })

        if progress_callback:
            progress_callback(i, total, f"Finished part {i} of {total}.")

    duration = 0.0
    if chunks:
        last = chunks[-1]
        duration = float(last.get("offset", 0.0)) + float(last.get("duration", 0.0))

    return {
        "text": "\n".join(text_parts).strip(),
        "segments": all_segments,
        "duration": duration,
        "model": MODEL,
    }
