from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import BinaryIO, Optional

import requests

TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
TRANSLATION_MODEL = "whisper-large-v3"
BASE_URL = "https://api.groq.com/openai/v1/audio"
DIRECT_UPLOAD_LIMIT = 24 * 1024 * 1024  # stay safely below Groq free-tier 25 MB attachment cap


def _mime_type(filename: str, supplied: Optional[str] = None) -> str:
    if supplied and supplied != "application/octet-stream":
        return supplied
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def _request_with_retry(method, url: str, *, headers: dict, timeout=(30, 600), **kwargs):
    last_error = None
    for attempt in range(4):
        try:
            response = method(url, headers=headers, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 3:
                raise RuntimeError(f"Network error while contacting Groq: {exc}") from exc
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 429 and attempt < 3:
            wait = response.headers.get("retry-after")
            try:
                delay = max(1.0, min(30.0, float(wait))) if wait else 2 ** attempt
            except (TypeError, ValueError):
                delay = 2 ** attempt
            time.sleep(delay)
            continue

        return response

    raise RuntimeError(f"Could not contact Groq: {last_error}")


def transcribe_upload(
    uploaded_file: BinaryIO,
    filename: str,
    size_bytes: int,
    api_key: str,
    language: Optional[str] = None,
    translate: bool = False,
    content_type: Optional[str] = None,
) -> dict:
    """Transcribe without local decoding/transcoding.

    Files below Groq's attachment limit are streamed as multipart uploads.
    Larger files are sent through Groq's documented URL field as a Base64 data URL,
    avoiding FFmpeg and CPU-heavy preprocessing on Streamlit Community Cloud.
    """
    if not api_key:
        raise ValueError("A Groq API key is required.")

    endpoint = "translations" if translate else "transcriptions"
    model = TRANSLATION_MODEL if translate else TRANSCRIPTION_MODEL
    url = f"{BASE_URL}/{endpoint}"
    headers = {"Authorization": f"Bearer {api_key}"}

    common = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": 0,
    }
    if language and not translate:
        common["language"] = language

    mime = _mime_type(filename, content_type)
    uploaded_file.seek(0)

    if size_bytes <= DIRECT_UPLOAD_LIMIT:
        files = {"file": (Path(filename).name, uploaded_file, mime)}
        form = {key: str(value) for key, value in common.items()}
        response = _request_with_retry(
            requests.post,
            url,
            headers=headers,
            data=form,
            files=files,
        )
        transfer_mode = "direct"
    else:
        # Groq documents the `url` parameter for files over the attachment limit
        # and supports Base64URL. A data URL keeps the file private and avoids
        # hosting it elsewhere just to obtain a public URL.
        raw = uploaded_file.read()
        encoded = base64.b64encode(raw).decode("ascii")
        data_url = f"data:{mime};base64,{encoded}"
        payload = dict(common)
        payload["url"] = data_url
        response = _request_with_retry(
            requests.post,
            url,
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
        )
        transfer_mode = "base64-url"
        del raw, encoded, data_url, payload

    if not response.ok:
        try:
            detail = response.json()
        except Exception:
            detail = response.text[-1500:]
        raise RuntimeError(
            f"Groq returned HTTP {response.status_code}: {detail}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Groq returned an unreadable transcription response.") from exc

    text = str(payload.get("text", "") or "").strip()
    segments = []
    for seg in payload.get("segments") or []:
        seg_text = str(seg.get("text", "") or "").strip()
        if not seg_text:
            continue
        segments.append(
            {
                "start": float(seg.get("start", 0.0) or 0.0),
                "end": float(seg.get("end", 0.0) or 0.0),
                "text": seg_text,
            }
        )

    duration = float(payload.get("duration", 0.0) or 0.0)
    if duration <= 0 and segments:
        duration = max(seg["end"] for seg in segments)

    if not segments and text:
        segments = [{"start": 0.0, "end": duration, "text": text}]

    return {
        "text": text,
        "segments": segments,
        "duration": duration,
        "model": model,
        "translation": translate,
        "transfer_mode": transfer_mode,
    }
