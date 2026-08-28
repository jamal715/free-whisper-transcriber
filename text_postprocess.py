from __future__ import annotations

import re
import time
from typing import Callable, Optional

import requests

CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
TRANSLATION_MODEL = "qwen/qwen3.8-27b"
SUMMARY_MODEL = "local-extractive-v1"


def _chat(*, api_key: str, model: str, system: str, user: str, max_retries: int = 6) -> str:
    if not api_key:
        raise ValueError("A Groq API key is required.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "reasoning_effort": "none",
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=(30, 240))
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Network error during text processing: {exc}") from exc
            time.sleep(min(20, 2 ** attempt))
            continue

        if response.status_code == 429 and attempt < max_retries - 1:
            retry_after = response.headers.get("retry-after")
            try:
                wait = float(retry_after) if retry_after else min(45, 4 * (attempt + 1))
            except (TypeError, ValueError):
                wait = min(45, 4 * (attempt + 1))
            time.sleep(max(2.0, min(60.0, wait)))
            continue

        if response.status_code in {408, 409, 500, 502, 503, 504} and attempt < max_retries - 1:
            time.sleep(min(20, 2 ** attempt))
            continue

        if not response.ok:
            try:
                detail = response.json()
            except Exception:
                detail = response.text[-1000:]
            raise RuntimeError(f"Groq text model returned HTTP {response.status_code}: {detail}")

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Groq text model returned no completion.")
        text = str(choices[0].get("message", {}).get("content", "") or "").strip()
        if not text:
            raise RuntimeError("Groq text model returned an empty completion.")
        return text

    raise RuntimeError("Text processing could not complete after retries.")


def translate_validated_chunks(
    *,
    validated_chunks: list[dict],
    api_key: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Translate only high-confidence source windows.

    REVIEW/FAILED windows are withheld instead of being turned into fluent but
    potentially misleading English.
    """
    system = (
        "Translate Pakistani Urdu-English code-switched research speech into faithful English. "
        "Preserve names, institutions, numbers, units, dates, amounts and technical terms. "
        "Do not summarize or infer. If something is unclear, write [unclear]. Return only the translation."
    )

    out = []
    total = len(validated_chunks)

    for i, chunk in enumerate(validated_chunks, start=1):
        source = str(chunk.get("selected_text", "") or "").strip()
        state = str(chunk.get("status", "review") or "review").lower()
        if progress_callback:
            progress_callback(i - 1, total, f"Translating part {i} of {total}…")

        if state == "silence":
            translated = "[silence / no confident speech]"
        elif state == "failed":
            translated = "[REVIEW REQUIRED — source transcription failed validation; translation withheld.]"
        elif state == "review":
            translated = "[VERIFY SOURCE AUDIO — translation withheld because source confidence is low.]"
        elif not source:
            translated = "[REVIEW REQUIRED — no reliable source transcript was available.]"
        else:
            translated = _chat(
                api_key=api_key,
                model=TRANSLATION_MODEL,
                system=system,
                user=source,
            )

        out.append({
            "index": i,
            "start": float(chunk.get("start", 0.0) or 0.0),
            "end": float(chunk.get("end", 0.0) or 0.0),
            "text": translated,
            "source_status": state,
            "source_score": chunk.get("score"),
        })
        if progress_callback:
            progress_callback(i, total, f"Translated part {i} of {total}.")

    return {
        "model": TRANSLATION_MODEL,
        "chunks": out,
        "text": "\n\n".join(c["text"] for c in out if c["text"]).strip(),
    }


def _fmt_time(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _compact_excerpt(text: str, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind(". "), cut.rfind("؟ "), cut.rfind("? "), cut.rfind("! "))
    if boundary > int(limit * 0.55):
        cut = cut[: boundary + 1]
    return cut.rstrip() + "…"


def build_research_summary(*, validated_chunks: list[dict], api_key: str = "") -> dict:
    """Create an instant, non-generative research digest.

    The function copies representative excerpts from PASSED windows only. It
    makes no model/API call, so it is fast and cannot add facts that were not in
    the validated transcript.
    """
    passed = [
        c for c in validated_chunks
        if str(c.get("status", "")).lower() == "passed"
        and str(c.get("selected_text", "") or "").strip()
    ]
    review_count = sum(str(c.get("status", "")).lower() == "review" for c in validated_chunks)
    failed_count = sum(str(c.get("status", "")).lower() == "failed" for c in validated_chunks)
    silence_count = sum(str(c.get("status", "")).lower() == "silence" for c in validated_chunks)

    if not passed:
        return {
            "english": "No source-grounded summary is available because no transcript windows passed validation.",
            "roman_urdu": "",
            "model": SUMMARY_MODEL,
        }

    # Choose representative windows across the whole recording rather than only
    # the beginning. At most 12 excerpts keeps the digest executive and quick.
    target = min(12, len(passed))
    if target == 1:
        chosen = [passed[0]]
    else:
        indices = sorted({round(i * (len(passed) - 1) / (target - 1)) for i in range(target)})
        chosen = [passed[i] for i in indices]

    bullets = []
    for c in chosen:
        start = _fmt_time(float(c.get("start", 0.0) or 0.0))
        end = _fmt_time(float(c.get("end", 0.0) or 0.0))
        excerpt = _compact_excerpt(str(c.get("selected_text", "") or ""))
        bullets.append(f"- **{start}–{end}** — {excerpt}")

    note = (
        f"Source-grounded extractive digest from {len(passed)} passed window(s). "
        f"Excluded: {review_count} review, {failed_count} failed, {silence_count} silence. "
        "No generative facts were added."
    )

    return {
        "english": note + "\n\n" + "\n".join(bullets),
        "roman_urdu": "",
        "model": SUMMARY_MODEL,
    }
