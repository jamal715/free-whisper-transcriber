from __future__ import annotations

import time
from typing import Callable, Optional

import requests

TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
TRANSLATION_MODEL = "whisper-large-v3"
BASE_URL = "https://api.groq.com/openai/v1/audio"


def _request_chunk(
    path: str,
    api_key: str,
    language: Optional[str],
    translate: bool,
    model: str,
) -> dict:
    endpoint = "translations" if translate else "transcriptions"
    url = f"{BASE_URL}/{endpoint}"

    data = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": "0",
    }
    if language and not translate:
        data["language"] = language

    headers = {"Authorization": f"Bearer {api_key}"}

    for attempt in range(4):
        with open(path, "rb") as audio_file:
            files = {"file": ("audio.mp3", audio_file, "audio/mpeg")}
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=(30, 300),
                )
            except requests.RequestException as exc:
                if attempt == 3:
                    raise RuntimeError(f"Network error while contacting transcription service: {exc}") from exc
                time.sleep(2 ** attempt)
                continue

        if response.status_code == 429 and attempt < 3:
            wait = response.headers.get("retry-after")
            try:
                delay = max(1.0, min(30.0, float(wait))) if wait else 2 ** attempt
            except ValueError:
                delay = 2 ** attempt
            time.sleep(delay)
            continue

        if not response.ok:
            try:
                detail = response.json()
            except Exception:
                detail = response.text[-1000:]
            raise RuntimeError(
                f"Transcription service returned HTTP {response.status_code}: {detail}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("Transcription service returned an unreadable response.") from exc

    raise RuntimeError("Transcription service could not complete the request.")


def transcribe_chunks(
    chunks: list[dict],
    api_key: str,
    language: Optional[str] = None,
    translate: bool = False,
    model: str = TRANSCRIPTION_MODEL,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    if not api_key:
        raise ValueError("A Groq API key is required.")

    all_segments = []
    full_text_parts = []
    total = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):
        if progress_callback:
            progress_callback(
                idx - 1,
                total,
                f"Transcribing chunk {idx} of {total}…",
            )

        payload = _request_chunk(
            path=chunk["path"],
            api_key=api_key,
            language=language,
            translate=translate,
            model=model,
        )

        text = str(payload.get("text", "") or "").strip()
        if text:
            full_text_parts.append(text)

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
            progress_callback(
                idx,
                total,
                f"Finished chunk {idx} of {total}.",
            )

    return {
        "text": "\n".join(full_text_parts).strip(),
        "segments": all_segments,
        "model": model,
        "translation": translate,
    }
