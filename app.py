from pathlib import Path
import shutil
import tempfile

import streamlit as st

from media import prepare_audio, probe_duration
from transcriber import MODEL_OPTIONS, choose_model, transcribe_file
from summarizer import extractive_summary
from utils import transcript_to_srt, format_seconds


st.set_page_config(
    page_title="Whisper Transcriber",
    page_icon="🎙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 0.25rem 0 1.1rem 0;
    }
    .hero h1 {
        margin-bottom: 0.15rem;
    }
    .soft-card {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin: .5rem 0 1rem 0;
    }
    .small-note {
        opacity: .75;
        font-size: .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🎙️ Whisper Transcriber</h1>
      <div class="small-note">
        Private-by-design processing on the app server · no paid transcription API
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Transcription")
    mode = st.selectbox(
        "Performance mode",
        list(MODEL_OPTIONS),
        index=0,
        help=(
            "Smart is recommended on Streamlit Community Cloud. "
            "For long recordings it automatically uses the lighter Whisper model."
        ),
    )

    language_label = st.selectbox(
        "Language",
        ["Auto detect", "English", "Urdu"],
        index=0,
        help="Auto detect is recommended for mixed Urdu/English recordings.",
    )
    language_map = {"Auto detect": None, "English": "en", "Urdu": "ur"}

    task_label = st.selectbox(
        "Output language",
        ["Keep spoken language", "Translate to English"],
        index=0,
    )
    task = "transcribe" if task_label == "Keep spoken language" else "translate"

    timestamps = st.toggle("Timestamps in TXT", value=True)

    st.divider()
    st.caption(
        "Cloud-safe mode is always on: 1 CPU thread, int8 inference, "
        "greedy decoding and silence skipping."
    )

uploaded = st.file_uploader(
    "Drop an audio or video recording",
    type=["mp3", "mp4", "m4a", "wav", "webm", "mpeg", "mpga", "ogg", "flac"],
    accept_multiple_files=False,
    help="For fastest uploads, audio-only MP3/M4A is better than MP4 video.",
)

if uploaded is None:
    st.markdown(
        """
        <div class="soft-card">
          <b>How it works</b><br>
          1. Upload a recording &nbsp;→&nbsp; 2. Audio is normalized &nbsp;→&nbsp;
          3. Whisper transcribes &nbsp;→&nbsp; 4. Download TXT/SRT
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Ready for your recording.")
    st.stop()

suffix = Path(uploaded.name).suffix.lower() or ".media"
file_size_mb = uploaded.size / (1024 * 1024)

c1, c2, c3 = st.columns(3)
c1.metric("File", uploaded.name)
c2.metric("Upload size", f"{file_size_mb:.1f} MB")
c3.metric("Engine", "Cloud-safe Whisper")

if suffix in {".mp4", ".webm", ".mpeg"}:
    st.caption(
        "Video preview is intentionally disabled to save memory. "
        "Only the audio track is used for transcription."
    )

if st.button("Transcribe recording", type="primary", use_container_width=True):
    st.session_state.pop("result", None)
    st.session_state["source_name"] = uploaded.name

    original_path = None
    audio_path = None

    progress = st.progress(0.0, text="Preparing upload…")
    status = st.empty()

    def set_progress(value: float, message: str):
        progress.progress(max(0.0, min(1.0, value)), text=message)
        status.caption(message)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            original_path = Path(tmp.name)
            uploaded.seek(0)
            shutil.copyfileobj(uploaded, tmp, length=1024 * 1024)

        set_progress(0.03, "Extracting and optimizing audio…")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_tmp:
            audio_path = Path(audio_tmp.name)

        prepare_audio(str(original_path), str(audio_path))
        try:
            original_path.unlink(missing_ok=True)
            original_path = None
        except Exception:
            pass

        duration = probe_duration(str(audio_path))
        selected_model = choose_model(mode, duration)

        set_progress(
            0.07,
            f"Audio ready ({format_seconds(duration)}). "
            f"Loading {selected_model.title()} model…",
        )

        def on_transcription_progress(value: float, message: str):
            # Reserve first 8% for upload/audio preparation.
            mapped = 0.08 + (0.91 * value)
            set_progress(mapped, message)

        result = transcribe_file(
            str(audio_path),
            model_id=selected_model,
            language=language_map[language_label],
            task=task,
            progress_callback=on_transcription_progress,
        )
        result["source_name"] = uploaded.name
        result["duration"] = duration or result.get("duration", 0.0)
        st.session_state["result"] = result

        progress.progress(1.0, text="Transcription complete")
        status.empty()
    except Exception as exc:
        st.error("The transcription could not finish.")
        st.exception(exc)
        st.info(
            "If this was a very long recording, retry with **Fastest** mode. "
            "The app is already restricted to one CPU thread to avoid Community Cloud throttling."
        )
    finally:
        for path in (original_path, audio_path):
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass

result = st.session_state.get("result")
if not result:
    st.stop()

segments = result["segments"]
detected_language = str(result.get("language", "unknown"))
duration = float(result.get("duration", 0.0))
model_id = result.get("model", "unknown")
source_name = result.get("source_name", st.session_state.get("source_name", "recording"))

st.success(
    f"Transcription complete · {format_seconds(duration)} · "
    f"language: {detected_language.upper()} · model: {model_id}"
)

plain_text = "\n".join(
    seg["text"].strip() for seg in segments if seg["text"].strip()
)

if timestamps:
    txt_text = "\n".join(
        f"[{format_seconds(seg['start'])} → {format_seconds(seg['end'])}] "
        f"{seg['text'].strip()}"
        for seg in segments
        if seg["text"].strip()
    )
else:
    txt_text = plain_text

srt_text = transcript_to_srt(segments)
stem = Path(source_name).stem

tab1, tab2, tab3 = st.tabs(["Transcript", "Quick summary", "Downloads"])

with tab1:
    st.text_area(
        "Transcript",
        value=txt_text,
        height=520,
        label_visibility="collapsed",
    )

with tab2:
    summary_length = st.slider(
        "Summary length",
        min_value=3,
        max_value=20,
        value=8,
    )
    summary = extractive_summary(plain_text, max_sentences=summary_length)
    if summary:
        st.markdown(summary)
        st.caption(
            "Free extractive summary — no external LLM/API call is made."
        )
    else:
        st.info("Not enough transcript text to create a summary.")

with tab3:
    summary = extractive_summary(plain_text, max_sentences=8)
    st.download_button(
        "⬇️ Transcript (.txt)",
        data=txt_text.encode("utf-8"),
        file_name=f"{stem}_transcript.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.download_button(
        "⬇️ Subtitles (.srt)",
        data=srt_text.encode("utf-8"),
        file_name=f"{stem}_transcript.srt",
        mime="application/x-subrip",
        use_container_width=True,
    )
    st.download_button(
        "⬇️ Quick summary (.txt)",
        data=summary.encode("utf-8"),
        file_name=f"{stem}_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )

st.caption(
    "Uploaded media is used temporarily for processing and the app deletes its temporary files afterwards."
)
