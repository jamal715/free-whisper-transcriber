from __future__ import annotations

import os
from typing import Callable, Optional

# Prevent BLAS/OpenMP libraries from spawning extra workers on free cloud CPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import streamlit as st
from faster_whisper import WhisperModel


MODEL_OPTIONS = {
    "Smart — recommended": "auto",
    "Fastest — long recordings": "tiny",
    "Balanced — better accuracy": "base",
}


def choose_model(mode_name: str, duration_seconds: float) -> str:
    selected = MODEL_OPTIONS.get(mode_name, "auto")
    if selected != "auto":
        return selected

    # Streamlit Community Cloud: prefer tiny for long recordings to stay
    # within shared CPU/RAM constraints; use base for shorter clips.
    if duration_seconds >= 20 * 60:
        return "tiny"
    return "base"


@st.cache_resource(show_spinner=False, max_entries=1)
def load_model(model_id: str) -> WhisperModel:
    return WhisperModel(
        model_id,
        device="cpu",
        compute_type="int8",
        cpu_threads=1,
        num_workers=1,
    )


def transcribe_file(
    file_path: str,
    model_id: str,
    language: Optional[str] = None,
    task: str = "transcribe",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    if progress_callback:
        progress_callback(0.01, f"Loading Whisper {model_id}…")

    model = load_model(model_id)

    if progress_callback:
        progress_callback(0.03, "Detecting speech and language…")

    segments_iter, info = model.transcribe(
        file_path,
        language=language,
        task=task,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        },
        condition_on_previous_text=False,
        word_timestamps=False,
    )

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    segments = []

    for seg in segments_iter:
        text = (seg.text or "").strip()
        if text:
            segments.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": text,
                }
            )

        if progress_callback:
            if duration > 0:
                ratio = min(0.99, max(0.04, float(seg.end) / duration))
            else:
                ratio = 0.5
            progress_callback(
                ratio,
                f"Transcribing… {format_seconds_short(float(seg.end))}"
                + (f" / {format_seconds_short(duration)}" if duration else ""),
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
