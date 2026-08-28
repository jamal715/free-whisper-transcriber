from __future__ import annotations

import mimetypes
import time
from pathlib import Path

import requests

OPENAI_URL = "https://api.openai.com/v1/audio/transcriptions"
PRIMARY_MODEL = "gpt-transcribe"
FALLBACK_MODEL = "gpt-4o-transcribe"


def _request(*, path: str, api_key: str, model: str, context_prompt: str = "") -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": model}
    if context_prompt:
        data["prompt"] = context_prompt[:900]

    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"

    for attempt in range(5):
        try:
            with open(path, "rb") as audio:
                response = requests.post(
                    OPENAI_URL,
                    headers=headers,
                    data=data,
                    files={"file": (Path(path).name, audio, mime)},
                    timeout=(30, 900),
                )
        except requests.RequestException as exc:
            if attempt == 4:
                raise RuntimeError(f"Network error while contacting OpenAI: {exc}") from exc
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

        if response.status_code == 400 and context_prompt:
            data.pop("prompt", None)
            context_prompt = ""
            continue

        if not response.ok:
            try:
                detail = response.json()
            except Exception:
                detail = response.text[-1000:]
            raise RuntimeError(f"OpenAI returned HTTP {response.status_code}: {detail}")

        payload = response.json()
        text = str(payload.get("text", "") or "").strip()
        if not text:
            raise RuntimeError("OpenAI returned an empty transcription.")
        return text

    raise RuntimeError("OpenAI could not complete the transcription request.")


def transcribe_for_validation(*, path: str, api_key: str, context_prompt: str = "") -> dict:
    if not api_key:
        raise ValueError("An OpenAI API key is required for dual-engine validation.")

    try:
        text = _request(
            path=path,
            api_key=api_key,
            model=PRIMARY_MODEL,
            context_prompt=context_prompt,
        )
        return {"text": text, "model": PRIMARY_MODEL}
    except RuntimeError as first_error:
        try:
            text = _request(
                path=path,
                api_key=api_key,
                model=FALLBACK_MODEL,
                context_prompt=context_prompt,
            )
            return {"text": text, "model": FALLBACK_MODEL, "fallback_reason": str(first_error)}
        except Exception:
            raise first_error
