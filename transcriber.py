from __future__ import annotations

from typing import Callable, Optional

import streamlit as st
from faster_whisper import WhisperModel


MODEL_OPTIONS = {
    "Tiny — fastest": "tiny",
    "Base — recommended": "base",
    "Small — more accurate": "small",
}


@st.cache_resource(show_spinner=False)
def load_model(model_id: str) -> WhisperModel:
    return WhisperModel(
        model_id,
        device="cpu",
        compute_type="int8",
    )


def transcribe_file(
    file_path: str,
    model_name: str = "Base — recommended",
    language: Optional[str] = None,
    task: str = "transcribe",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    model_id = MODEL_OPTIONS[model_name]

    if progress_callback:
        progress_callback(0.02, f"Loading Whisper {model_id} model…")

    model = load_model(model_id)

    if progress_callback:
        progress_callback(0.06, "Analyzing recording…")

    segments_iter, info = model.transcribe(
        file_path,
        language=language,
        task=task,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=True,
    )

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    segments = []

    for seg in segments_iter:
        segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text,
            }
        )

        if progress_callback:
            if duration > 0:
                ratio = min(0.98, 0.06 + 0.92 * (float(seg.end) / duration))
            else:
                ratio = 0.5
            progress_callback(
                ratio,
                f"Transcribing… {format_seconds_short(float(seg.end))}",
            )

    return {
        "segments": segments,
        "language": getattr(info, "language", language or "unknown"),
        "language_probability": float(
            getattr(info, "language_probability", 0.0) or 0.0
        ),
        "duration": duration or (segments[-1]["end"] if segments else 0.0),
        "model": model_id,
        "task": task,
    }


def format_seconds_short(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
