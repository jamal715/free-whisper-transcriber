from __future__ import annotations

from difflib import SequenceMatcher
import re
import unicodedata

_ALLOWED_SCRIPT_RANGES = (
    (0x0000, 0x024F),   # Latin + punctuation
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic supplement
    (0x08A0, 0x08FF),   # Arabic extended
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


def transcript_similarity(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return SequenceMatcher(None, wa, wb, autojunk=False).ratio()


def inspect_text(text: str) -> dict:
    text = (text or "").strip()
    reasons: list[str] = []
    if not text:
        return {"score": 0, "status": "failed", "reasons": ["empty transcript"]}

    visible = [c for c in text if not c.isspace()]
    words = _words(text)
    weird = [c for c in visible if not _is_allowed_char(c)]
    weird_ratio = len(weird) / max(1, len(visible))

    digits = sum(c.isdigit() for c in visible)
    digit_ratio = digits / max(1, len(visible))

    alpha = sum(c.isalpha() for c in visible)
    alpha_ratio = alpha / max(1, len(visible))

    if weird_ratio > 0.025:
        reasons.append("unexpected script/characters")
    if digit_ratio > 0.42 and len(visible) > 80:
        reasons.append("abnormally number-heavy output")
    if alpha_ratio < 0.35 and len(visible) > 80:
        reasons.append("too little recognizable language")

    if len(words) >= 20:
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        max_share = max(counts.values()) / len(words)
        if max_share > 0.30:
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
        "extreme token repetition": 28,
        "long numeric sequence": 40,
        "repeated-character corruption": 30,
    }
    for reason in reasons:
        score -= penalties.get(reason, 15)
    score = max(0, score)

    if score >= 85:
        status = "passed"
    elif score >= 60:
        status = "review"
    else:
        status = "failed"

    return {
        "score": score,
        "status": status,
        "reasons": reasons,
        "weird_ratio": round(weird_ratio, 4),
        "digit_ratio": round(digit_ratio, 4),
        "alpha_ratio": round(alpha_ratio, 4),
    }


def assess_dual(groq_text: str, verifier_text: str | None) -> dict:
    primary = inspect_text(groq_text)
    if not verifier_text:
        return {
            **primary,
            "similarity": None,
            "dual_verified": False,
            "selected_provider": "Groq Whisper Large V3",
            "selected_text": groq_text,
        }

    verifier = inspect_text(verifier_text)
    similarity = transcript_similarity(groq_text, verifier_text)

    if primary["status"] == "failed" and verifier["status"] == "passed":
        selected_text = verifier_text
        selected_provider = "OpenAI GPT-Transcribe rescue"
        score = max(72, verifier["score"] - 8)
        reasons = list(primary["reasons"]) + ["Groq output replaced after independent rescue pass"]
        status = "review" if similarity < 0.45 else "passed"
    else:
        selected_text = groq_text
        selected_provider = "Groq Whisper Large V3"
        score = min(primary["score"], verifier["score"])
        reasons = list(dict.fromkeys(primary["reasons"] + verifier["reasons"]))
        if similarity >= 0.78 and primary["status"] == "passed" and verifier["status"] == "passed":
            score = max(score, 94)
            status = "passed"
        elif similarity >= 0.62 and primary["status"] != "failed" and verifier["status"] != "failed":
            score = min(score, 82)
            status = "review"
            reasons.append("independent engines differ moderately")
        else:
            score = min(score, 52)
            status = "failed"
            reasons.append("independent engines disagree strongly")

    return {
        "score": max(0, min(100, int(round(score)))),
        "status": status,
        "reasons": list(dict.fromkeys(reasons)),
        "similarity": round(similarity, 3),
        "dual_verified": True,
        "selected_provider": selected_provider,
        "selected_text": selected_text,
        "groq_quality": primary,
        "verifier_quality": verifier,
    }


def overall_health(validated_chunks: list[dict]) -> dict:
    if not validated_chunks:
        return {"score": 0, "status": "failed", "passed": 0, "review": 0, "failed": 0}
    scores = [int(c.get("score", 0)) for c in validated_chunks]
    passed = sum(c.get("status") == "passed" for c in validated_chunks)
    review = sum(c.get("status") == "review" for c in validated_chunks)
    failed = sum(c.get("status") == "failed" for c in validated_chunks)
    score = round(sum(scores) / len(scores))
    if failed:
        status = "needs review"
    elif review:
        status = "good with review"
    else:
        status = "high confidence"
    return {"score": score, "status": status, "passed": passed, "review": review, "failed": failed}
