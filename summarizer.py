from __future__ import annotations

import re
from collections import Counter


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "here", "him", "his", "i", "if", "in", "into", "is", "it",
    "its", "just", "me", "more", "my", "no", "not", "of", "on", "or", "our",
    "out", "so", "some", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "up", "us", "very", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "would", "you",
    "your", "yeah", "yes", "okay", "ok", "um", "uh",
}


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = re.split(r"(?<=[.!?؟])\s+|\n+", text)
    chunks = [s.strip() for s in chunks if len(s.strip()) >= 25]

    if len(chunks) <= 1:
        words = text.split()
        chunks = [
            " ".join(words[i:i + 35])
            for i in range(0, len(words), 35)
            if len(words[i:i + 35]) >= 8
        ]
    return chunks


def extractive_summary(text: str, max_sentences: int = 8) -> str:
    sentences = _sentences(text)
    if not sentences:
        return ""

    words = re.findall(r"[\w\u0600-\u06FF']+", text.lower(), flags=re.UNICODE)
    frequencies = Counter(
        word for word in words
        if len(word) > 2 and word not in STOPWORDS and not word.isdigit()
    )

    if not frequencies:
        selected = sentences[:max_sentences]
    else:
        max_freq = max(frequencies.values())
        scores = []
        for idx, sentence in enumerate(sentences):
            sentence_words = re.findall(
                r"[\w\u0600-\u06FF']+",
                sentence.lower(),
                flags=re.UNICODE,
            )
            score = sum(
                frequencies[word] / max_freq
                for word in sentence_words
                if word in frequencies
            )
            score = score / max(1.0, len(sentence_words) ** 0.45)
            scores.append((score, idx, sentence))

        best = sorted(scores, reverse=True)[:max_sentences]
        selected = [item[2] for item in sorted(best, key=lambda x: x[1])]

    return "\n".join(f"- {sentence}" for sentence in selected)
