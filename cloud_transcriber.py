from __future__ import annotations

import mimetypes
import time
from pathlib import Path
from typing import Callable, Optional

import requests

ACCURACY_MODEL = "whisper-large-v3"
FAST_MODEL = "whisper-large-v3-turbo"
TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
TRANSLATION_URL = "https://api.groq.com/openai/v1/audio/translations"

LOGPROB_REVIEW_THRESHOLD = -1.0
NO_SPEECH_REVIEW_THRESHOLD = 0.6
COMPRESSION_REVIEW_THRESHOLD = 2.4


def _compact_prompt(context: str, previous_tail: str = "") -> str:
    parts = []
    context = (context or "").strip()
    previous_tail = (previous_tail or "").strip()
    if context:
        parts.append(context[:600])
    if previous_tail:
        parts.append("Previous transcript ending: " + previous_tail[-280:])
    return "\n".join(parts)[:850]


def _post_audio(
    *,
    path: str,
    api_key: str,
    endpoint: str,
    model: str,
    language: Optional[str] = None,
    prompt: str = "",
) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": "0",
    }
    if language and endpoint == TRANSCRIPTION_URL:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt

    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"

    for attempt in range(5):
        try:
            with open(path, "rb") as audio:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    data=data,
                    files={"file": (Path(path).name, audio, mime)},
                    timeout=(30, 900),
                )
        except requests.RequestException as exc:
            if attempt == 4:
                raise RuntimeError(f"Network error while contacting Groq: {exc}") from exc
            time.sleep(min(16, 2 ** attempt))
            continue

        if response.status_code in {408, 409, 429, 500, 502, 503, 504} and attempt < 4:
            retry_after = response.headers.get("retry-after")
            try:
                wait = float(retry_after) if retry_after else 2 ** attempt
            except (TypeError, ValueError):
                wait = 2 ** attempt
            time.sleep(max(1.0, min(30.0, wait)))
            continue

        if not response.ok:
            try:
                detail = response.json()
            except Exception:
                detail = response.text[-1200:]
            raise RuntimeError(f"Groq returned HTTP {response.status_code}: {detail}")

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("Groq returned an unreadable response.") from exc

    raise RuntimeError("Groq could not complete the request after retries.")


def _review_metadata(seg: dict) -> tuple[bool, list[str]]:
    reasons = []
    avg_logprob = seg.get("avg_logprob")
    if isinstance(avg_logprob, (int, float)) and avg_logprob < LOGPROB_REVIEW_THRESHOLD:
        reasons.append("low average log probability")

    no_speech = seg.get("no_speech_prob")
    if isinstance(no_speech, (int, float)) and no_speech > NO_SPEECH_REVIEW_THRESHOLD:
        reasons.append("high no-speech probability")

    compression = seg.get("compression_ratio")
    if isinstance(compression, (int, float)) and compression > COMPRESSION_REVIEW_THRESHOLD:
        reasons.append("unusual compression ratio")

    return bool(reasons), reasons


def _merge_payload(
    *,
    payload: dict,
    offset: float,
    all_segments: list[dict],
    text_parts: list[str],
    languages: list[str],
) -> str:
    text = str(payload.get("text", "") or "").strip()
    if text:
        text_parts.append(text)

    language = str(payload.get("language", "") or "").strip()
    if language:
        languages.append(language)

    raw_segments = payload.get("segments") or []
    if raw_segments:
        for seg in raw_segments:
            seg_text = str(seg.get("text", "") or "").strip()
            if not seg_text:
                continue
            review_flag, review_reasons = _review_metadata(seg)
            all_segments.append({
                "start": offset + float(seg.get("start", 0.0) or 0.0),
                "end": offset + float(seg.get("end", 0.0) or 0.0),
                "text": seg_text,
                "review_flag": review_flag,
                "review_reasons": review_reasons,
                "avg_logprob": seg.get("avg_logprob"),
                "no_speech_prob": seg.get("no_speech_prob"),
                "compression_ratio": seg.get("compression_ratio"),
            })
    elif text:
        all_segments.append({
            "start": offset,
            "end": offset + float(payload.get("duration", 0.0) or 0.0),
            "text": text,
            "review_flag": False,
            "review_reasons": [],
        })

    return text


def transcribe_chunks(
    *,
    chunks: list[dict],
    api_key: str,
    language: Optional[str] = None,
    model: str = ACCURACY_MODEL,
    context_prompt: str = "",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    if not api_key:
        raise ValueError("A Groq API key is required.")

    all_segments = []
    text_parts = []
    languages = []
    total = len(chunks)
    previous_tail = ""

    for i, chunk in enumerate(chunks, start=1):
        if progress_callback:
            progress_callback(i - 1, total, f"Transcribing part {i} of {total}…")

        payload = _post_audio(
            path=chunk["path"],
            api_key=api_key,
            endpoint=TRANSCRIPTION_URL,
            model=model,
            language=language,
            prompt=_compact_prompt(context_prompt, previous_tail),
        )

        text = _merge_payload(
            payload=payload,
            offset=float(chunk.get("offset", 0.0)),
            all_segments=all_segments,
            text_parts=text_parts,
            languages=languages,
        )
        if text:
            previous_tail = text[-350:]

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
        "model": model,
        "parts": total,
        "languages": languages,
    }


def translate_chunks(
    *,
    chunks: list[dict],
    api_key: str,
    context_prompt: str = "",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    if not api_key:
        raise ValueError("A Groq API key is required.")

    all_segments = []
    text_parts = []
    languages = []
    total = len(chunks)
    previous_tail = ""

    for i, chunk in enumerate(chunks, start=1):
        if progress_callback:
            progress_callback(i - 1, total, f"Translating part {i} of {total}…")

        payload = _post_audio(
            path=chunk["path"],
            api_key=api_key,
            endpoint=TRANSLATION_URL,
            model=ACCURACY_MODEL,
            prompt=_compact_prompt(context_prompt, previous_tail),
        )

        text = _merge_payload(
            payload=payload,
            offset=float(chunk.get("offset", 0.0)),
            all_segments=all_segments,
            text_parts=text_parts,
            languages=languages,
        )
        if text:
            previous_tail = text[-350:]

        if progress_callback:
            progress_callback(i, total, f"Finished translation part {i} of {total}.")

    duration = 0.0
    if chunks:
        last = chunks[-1]
        duration = float(last.get("offset", 0.0)) + float(last.get("duration", 0.0))

    return {
        "text": "\n".join(text_parts).strip(),
        "segments": all_segments,
        "duration": duration,
        "model": ACCURACY_MODEL,
        "parts": total,
        "languages": languages,
    }
