from __future__ import annotations

import time
from typing import Callable, Optional
import requests

CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
TRANSLATION_MODEL = "qwen/qwen3.8-27b"
SUMMARY_MODEL = "qwen/qwen3.8-27b"


def _chat(*, api_key: str, model: str, system: str, user: str, max_retries: int = 8) -> str:
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
        "temperature": 0.1,
        "reasoning_effort": "none",
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=(30, 300))
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Network error during text processing: {exc}") from exc
            time.sleep(min(30, 2 ** attempt))
            continue

        if response.status_code == 429 and attempt < max_retries - 1:
            retry_after = response.headers.get("retry-after")
            try:
                wait = float(retry_after) if retry_after else min(60, 5 * (attempt + 1))
            except (TypeError, ValueError):
                wait = min(60, 5 * (attempt + 1))
            time.sleep(max(2.0, min(90.0, wait)))
            continue

        if response.status_code in {408, 409, 500, 502, 503, 504} and attempt < max_retries - 1:
            time.sleep(min(30, 2 ** attempt))
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
    system = (
        "You are a professional research-interview translator. Translate Pakistani Urdu-English "
        "code-switched speech into faithful, natural English. Preserve every proper noun, institution, "
        "number, unit, date, amount, and technical term. Do not summarize, clean up meaning, or invent "
        "missing content. Keep already-English speech in English. If wording is genuinely unclear, use "
        "[unclear] rather than guessing. Return only the translation."
    )

    out = []
    total = len(validated_chunks)

    for i, chunk in enumerate(validated_chunks, start=1):
        source = str(chunk.get("selected_text", "") or "").strip()
        state = str(chunk.get("status", "review") or "review").lower()
        if progress_callback:
            progress_callback(i - 1, total, f"Translating validated part {i} of {total}…")

        if state == "failed":
            # Never turn a corrupted ASR result into polished-looking English.
            translated = "[REVIEW REQUIRED — source transcription failed validation; translation withheld.]"
        elif not source:
            translated = "[REVIEW REQUIRED — no reliable source transcript was available.]"
        else:
            translated = _chat(
                api_key=api_key,
                model=TRANSLATION_MODEL,
                system=system,
                user=source,
            )
            if state == "review":
                translated = "[VERIFY SOURCE AUDIO] " + translated

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


def build_research_summary(*, validated_chunks: list[dict], api_key: str) -> dict:
    if not validated_chunks:
        return {"english": "", "roman_urdu": "", "model": SUMMARY_MODEL}

    usable = [c for c in validated_chunks if str(c.get("status", "")).lower() != "failed"]
    failed_count = len(validated_chunks) - len(usable)
    if not usable:
        return {
            "english": "Summary withheld because all transcript sections failed validation.",
            "roman_urdu": "Summary roki gayi hai kyun ke tamam transcript sections validation mein fail huay.",
            "model": SUMMARY_MODEL,
        }

    chunk_notes = []
    system_notes = (
        "You are summarizing one VALIDATED part of a research interview. Extract only substantive claims, "
        "decisions, examples, numbers, institutions, disagreements, and unresolved questions. "
        "Do not infer beyond the transcript. Return 3-7 concise English bullets."
    )

    for chunk in usable:
        text = str(chunk.get("selected_text", "") or "").strip()
        if not text:
            continue
        note = _chat(
            api_key=api_key,
            model=SUMMARY_MODEL,
            system=system_notes,
            user=text[:12000],
        )
        chunk_notes.append(note)

    combined = "\n\n".join(chunk_notes)
    if not combined:
        return {"english": "", "roman_urdu": "", "model": SUMMARY_MODEL}

    caveat = ""
    if failed_count:
        caveat = (
            f"IMPORTANT: {failed_count} transcript section(s) failed validation and were excluded. "
            "State this limitation clearly in both summaries.\n\n"
        )

    system_final = (
        "You are producing a faithful research-interview summary from validated chunk-level notes. "
        "Do not add facts. First provide a concise English summary with clear bullets and, where useful, "
        "short section headings. Then provide a separate Roman Urdu summary conveying the same points "
        "in natural Pakistani Roman Urdu. Preserve names, numbers and institutions exactly. "
        "Use these exact headings: ENGLISH SUMMARY and ROMAN URDU SUMMARY."
    )
    final = _chat(
        api_key=api_key,
        model=SUMMARY_MODEL,
        system=system_final,
        user=(caveat + combined)[:50000],
    )

    english = final
    roman = ""
    marker = "ROMAN URDU SUMMARY"
    if marker in final:
        before, after = final.split(marker, 1)
        english = before.replace("ENGLISH SUMMARY", "", 1).strip()
        roman = after.strip()

    return {
        "english": english.strip(),
        "roman_urdu": roman.strip(),
        "model": SUMMARY_MODEL,
    }
