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
MIN_REQUEST_INTERVAL_SECONDS = 3.15

LOGPROB_REVIEW_THRESHOLD = -1.0
NO_SPEECH_REVIEW_THRESHOLD = 0.6
COMPRESSION_REVIEW_THRESHOLD = 2.4


def _compact_prompt(context: str, previous_tail: str = "") -> str:
    parts = []
    context = (context or "").strip()
    previous_tail = (previous_tail or "").strip()
    if context:
        parts.append(context[:500])
    if previous_tail:
        parts.append(previous_tail[-240:])
    return "\n".join(parts)[:720]


def _post_audio(*, path: str, api_key: str, endpoint: str, model: str, language: Optional[str] = None, prompt: str = "") -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": "0",
        "timestamp_granularities[]": "segment",
    }
    if language and endpoint == TRANSCRIPTION_URL:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt

    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    for attempt in range(6):
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
            if attempt == 5:
                raise RuntimeError(f"Network error while contacting Groq: {exc}") from exc
            time.sleep(min(20, 2 ** attempt))
            continue

        if response.status_code in {408, 409, 429, 500, 502, 503, 504} and attempt < 5:
            retry_after = response.headers.get("retry-after")
            try:
                wait = float(retry_after) if retry_after else 2 ** attempt
            except (TypeError, ValueError):
                wait = 2 ** attempt
            time.sleep(max(1.0, min(45.0, wait)))
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


def _payload_stats(payload: dict) -> dict:
    raw = payload.get("segments") or []
    no_speech = [float(s["no_speech_prob"]) for s in raw if isinstance(s.get("no_speech_prob"), (int, float))]
    logprobs = [float(s["avg_logprob"]) for s in raw if isinstance(s.get("avg_logprob"), (int, float))]
    return {
        "raw_segment_count": len(raw),
        "avg_no_speech_prob": (sum(no_speech) / len(no_speech)) if no_speech else None,
        "max_no_speech_prob": max(no_speech) if no_speech else None,
        "avg_logprob": (sum(logprobs) / len(logprobs)) if logprobs else None,
    }


def _merge_payload(*, payload: dict, offset: float, keep_after: float, all_segments: list[dict], text_parts: list[str], languages: list[str]) -> str:
    language = str(payload.get("language", "") or "").strip()
    if language:
        languages.append(language)

    raw_segments = payload.get("segments") or []
    kept_text: list[str] = []
    if raw_segments:
        for seg in raw_segments:
            seg_text = str(seg.get("text", "") or "").strip()
            if not seg_text:
                continue
            global_start = offset + float(seg.get("start", 0.0) or 0.0)
            global_end = offset + float(seg.get("end", 0.0) or 0.0)
            midpoint = (global_start + global_end) / 2.0
            if midpoint < keep_after - 0.05:
                continue
            review_flag, review_reasons = _review_metadata(seg)
            all_segments.append({
                "start": global_start,
                "end": global_end,
                "text": seg_text,
                "review_flag": review_flag,
                "review_reasons": review_reasons,
                "avg_logprob": seg.get("avg_logprob"),
                "no_speech_prob": seg.get("no_speech_prob"),
                "compression_ratio": seg.get("compression_ratio"),
            })
            kept_text.append(seg_text)
    else:
        text = str(payload.get("text", "") or "").strip()
        if text:
            all_segments.append({
                "start": max(offset, keep_after),
                "end": offset + float(payload.get("duration", 0.0) or 0.0),
                "text": text,
                "review_flag": False,
                "review_reasons": [],
            })
            kept_text.append(text)

    merged = " ".join(kept_text).strip()
    if merged:
        text_parts.append(merged)
    return merged


def transcribe_chunks(
    *,
    chunks: list[dict],
    api_key: str,
    language: Optional[str] = None,
    model: str = ACCURACY_MODEL,
    context_prompt: str = "",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    chunk_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    if not api_key:
        raise ValueError("A Groq API key is required.")

    all_segments: list[dict] = []
    text_parts: list[str] = []
    languages: list[str] = []
    chunk_results: list[dict] = []
    total = len(chunks)
    previous_tail = ""
    primary_failures = 0
    last_request_started = 0.0

    for i, chunk in enumerate(chunks, start=1):
        if progress_callback:
            progress_callback(i - 1, total, f"Listening to minute {i} of {total}…")

        elapsed = time.monotonic() - last_request_started
        if last_request_started and elapsed < MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        last_request_started = time.monotonic()

        payload = None
        primary_error = None
        try:
            payload = _post_audio(
                path=chunk["path"],
                api_key=api_key,
                endpoint=TRANSCRIPTION_URL,
                model=model,
                language=language,
                prompt=_compact_prompt(context_prompt, previous_tail),
            )
        except Exception as exc:
            primary_error = str(exc)
            primary_failures += 1

        nominal_end = float(chunk.get("nominal_end") or (float(chunk.get("offset", 0.0)) + float(chunk.get("duration", 0.0))))
        base_result = {
            "index": i,
            "offset": float(chunk.get("offset", 0.0)),
            "keep_after": float(chunk.get("keep_after", chunk.get("offset", 0.0))),
            "end": nominal_end,
            "duration": float(chunk.get("duration", 0.0)),
        }

        if payload is None:
            chunk_result = {
                **base_result,
                "text": "",
                "language": "",
                "segment_count": 0,
                "raw_segment_count": 0,
                "avg_no_speech_prob": None,
                "max_no_speech_prob": None,
                "avg_logprob": None,
                "error": primary_error,
            }
        else:
            before = len(all_segments)
            text = _merge_payload(
                payload=payload,
                offset=base_result["offset"],
                keep_after=base_result["keep_after"],
                all_segments=all_segments,
                text_parts=text_parts,
                languages=languages,
            )
            chunk_segments = all_segments[before:]
            stats = _payload_stats(payload)
            chunk_result = {
                **base_result,
                "text": text,
                "language": str(payload.get("language", "") or "").strip(),
                "segment_count": len(chunk_segments),
                "error": None,
                **stats,
            }
            if text:
                previous_tail = text[-260:]

        chunk_results.append(chunk_result)
        if chunk_callback:
            chunk_callback(chunk_result)
        if progress_callback:
            progress_callback(i, total, f"Minute {i} of {total} complete.")

    duration = max((float(c.get("nominal_end", 0.0) or 0.0) for c in chunks), default=0.0)
    return {
        "text": "\n".join(text_parts).strip(),
        "segments": all_segments,
        "duration": duration,
        "model": model,
        "parts": total,
        "languages": languages,
        "chunk_results": chunk_results,
        "primary_failures": primary_failures,
    }


def translate_chunks(*, chunks: list[dict], api_key: str, context_prompt: str = "", progress_callback: Optional[Callable[[int, int, str], None]] = None) -> dict:
    raise RuntimeError("Audio translation is disabled in free production mode. Text translation is used after validation instead.")
