from __future__ import annotations

import re
import unicodedata

_ALLOWED_SCRIPT_RANGES = (
    (0x0000, 0x024F),
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
)


def _is_allowed_char(ch: str) -> bool:
    if ch.isspace() or ch.isdigit():
        return True
    cp = ord(ch)
    if any(lo <= cp <= hi for lo, hi in _ALLOWED_SCRIPT_RANGES):
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF']+", text.lower())


def inspect_text(text: str) -> dict:
    text = (text or "").strip()
    reasons: list[str] = []
    if not text:
        return {"score": 0, "status": "failed", "reasons": ["empty transcript"]}

    visible = [c for c in text if not c.isspace()]
    words = _words(text)
    weird = [c for c in visible if not _is_allowed_char(c)]
    weird_ratio = len(weird) / max(1, len(visible))
    digit_ratio = sum(c.isdigit() for c in visible) / max(1, len(visible))
    alpha_ratio = sum(c.isalpha() for c in visible) / max(1, len(visible))

    if weird_ratio > 0.025:
        reasons.append("unexpected script/characters")
    if digit_ratio > 0.42 and len(visible) > 80:
        reasons.append("abnormally number-heavy output")
    if alpha_ratio < 0.35 and len(visible) > 80:
        reasons.append("too little recognizable language")

    if len(words) >= 20:
        counts: dict[str, int] = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        if max(counts.values()) / len(words) > 0.30:
            reasons.append("extreme token repetition")

    if re.search(r"(?:\b\d{2,}\b[\s,.;:/-]*){12,}", text):
        reasons.append("long numeric sequence")
    if re.search(r"(.)\1{8,}", text):
        reasons.append("repeated-character corruption")

    score = 100
    penalties = {
        "unexpected script/characters": 38,
        "abnormally number-heavy output": 30,
        "too little recognizable language": 30,
        "extreme token repetition": 34,
        "long numeric sequence": 45,
        "repeated-character corruption": 38,
    }
    for reason in reasons:
        score -= penalties.get(reason, 15)
    score = max(0, score)
    status = "passed" if score >= 85 else "review" if score >= 60 else "failed"

    return {
        "score": score,
        "status": status,
        "reasons": reasons,
        "weird_ratio": round(weird_ratio, 4),
        "digit_ratio": round(digit_ratio, 4),
        "alpha_ratio": round(alpha_ratio, 4),
    }


def assess_single(
    text: str,
    *,
    api_error: str | None = None,
    avg_no_speech_prob: float | None = None,
    max_no_speech_prob: float | None = None,
    segment_count: int = 0,
) -> dict:
    """Classify one Groq result without confusing silence with failure."""
    if api_error:
        return {
            "score": 0,
            "status": "failed",
            "reasons": ["transcription service error"],
            "selected_provider": "Groq Whisper Large V3",
            "selected_text": "",
        }

    clean = (text or "").strip()
    avg_ns = float(avg_no_speech_prob) if isinstance(avg_no_speech_prob, (int, float)) else None
    max_ns = float(max_no_speech_prob) if isinstance(max_no_speech_prob, (int, float)) else None

    likely_silence = (
        not clean
        and (
            segment_count == 0
            or (avg_ns is not None and avg_ns >= 0.55)
            or (max_ns is not None and max_ns >= 0.80)
        )
    )
    if likely_silence:
        return {
            "score": None,
            "status": "silence",
            "reasons": ["no confident speech detected"],
            "selected_provider": "Groq Whisper Large V3",
            "selected_text": "",
        }

    result = inspect_text(clean)
    result.update({
        "selected_provider": "Groq Whisper Large V3",
        "selected_text": clean,
    })
    return result


def assess_dual(groq_text: str, verifier_text: str | None) -> dict:
    # Kept for backwards compatibility; free production mode uses assess_single.
    primary = inspect_text(groq_text)
    return {
        **primary,
        "similarity": None,
        "dual_verified": False,
        "selected_provider": "Groq Whisper Large V3",
        "selected_text": groq_text,
    }


def overall_health(validated_chunks: list[dict]) -> dict:
    if not validated_chunks:
        return {"score": 0, "status": "failed", "passed": 0, "review": 0, "failed": 0, "silence": 0}

    speech_chunks = [c for c in validated_chunks if c.get("status") != "silence"]
    scores = [int(c.get("score", 0) or 0) for c in speech_chunks]
    passed = sum(c.get("status") == "passed" for c in validated_chunks)
    review = sum(c.get("status") == "review" for c in validated_chunks)
    failed = sum(c.get("status") == "failed" for c in validated_chunks)
    silence = sum(c.get("status") == "silence" for c in validated_chunks)

    score = round(sum(scores) / len(scores)) if scores else 100
    if failed:
        status = "needs review"
    elif review:
        status = "good with review"
    else:
        status = "high confidence"
    return {
        "score": score,
        "status": status,
        "passed": passed,
        "review": review,
        "failed": failed,
        "silence": silence,
    }
