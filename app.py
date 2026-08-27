from pathlib import Path
import tempfile

import streamlit as st

from transcriber import MODEL_OPTIONS, transcribe_file
from summarizer import extractive_summary
from utils import transcript_to_srt, format_seconds


st.set_page_config(
    page_title="Free Whisper Transcriber",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Free Whisper Transcriber")
st.caption(
    "Upload audio or video → transcribe with Whisper → download TXT/SRT. "
    "No paid transcription API required."
)

with st.sidebar:
    st.header("Transcription settings")
    model_name = st.selectbox(
        "Whisper model",
        list(MODEL_OPTIONS),
        index=1,
        help=(
            "Tiny = fastest. Base = best default for CPU. "
            "Small = more accurate but heavier/slower."
        ),
    )

    language_label = st.selectbox(
        "Language",
        ["Auto detect", "English", "Urdu"],
        index=0,
        help="Auto detect is usually best for mixed Urdu/English recordings.",
    )
    language_map = {
        "Auto detect": None,
        "English": "en",
        "Urdu": "ur",
    }

    task_label = st.selectbox(
        "Output",
        ["Transcribe in spoken language", "Translate speech to English"],
        index=0,
    )
    task = "transcribe" if task_label.startswith("Transcribe") else "translate"

    timestamps = st.checkbox("Include timestamps in TXT", value=True)

    st.divider()
    st.caption(
        "Tip: Start with Base. If names or technical terms are missed, retry with Small."
    )

uploaded = st.file_uploader(
    "Upload an audio or video file",
    type=["mp3", "mp4", "m4a", "wav", "webm", "mpeg", "mpga", "ogg", "flac"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Choose a recording above. Your transcript will appear here.")
    st.stop()

suffix = Path(uploaded.name).suffix or ".media"
file_size_mb = uploaded.size / (1024 * 1024)

c1, c2 = st.columns(2)
with c1:
    st.metric("File", uploaded.name)
with c2:
    st.metric("Size", f"{file_size_mb:.1f} MB")

if suffix.lower() in {".mp4", ".webm"}:
    st.video(uploaded)
else:
    st.audio(uploaded)

if st.button("Start transcription", type="primary", use_container_width=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        temp_path = Path(tmp.name)

    progress = st.progress(0.0, text="Preparing Whisper…")
    status = st.empty()

    def on_progress(value: float, message: str):
        progress.progress(max(0.0, min(1.0, value)), text=message)
        status.caption(message)

    try:
        result = transcribe_file(
            str(temp_path),
            model_name=model_name,
            language=language_map[language_label],
            task=task,
            progress_callback=on_progress,
        )
        st.session_state["result"] = result
        progress.progress(1.0, text="Transcription complete")
        status.empty()
    except Exception as exc:
        st.error(f"Transcription failed: {exc}")
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

result = st.session_state.get("result")
if not result:
    st.stop()

segments = result["segments"]
detected_language = result["language"]
duration = result["duration"]

st.success(
    f"Done — detected language: {detected_language.upper()} · "
    f"duration: {format_seconds(duration)}"
)

plain_text = "\n".join(seg["text"].strip() for seg in segments if seg["text"].strip())

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
        "Summary length (sentences/excerpts)",
        min_value=3,
        max_value=20,
        value=8,
    )
    summary = extractive_summary(plain_text, max_sentences=summary_length)
    if summary:
        st.markdown(summary)
        st.caption(
            "This is a free extractive summary: it selects important excerpts "
            "from the transcript rather than calling a paid AI API."
        )
    else:
        st.info("Not enough transcript text to build a summary.")

with tab3:
    stem = Path(uploaded.name).stem
    st.download_button(
        "Download transcript (.txt)",
        data=txt_text.encode("utf-8"),
        file_name=f"{stem}_transcript.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.download_button(
        "Download subtitles (.srt)",
        data=srt_text.encode("utf-8"),
        file_name=f"{stem}_transcript.srt",
        mime="application/x-subrip",
        use_container_width=True,
    )
    summary = extractive_summary(plain_text, max_sentences=8)
    st.download_button(
        "Download quick summary (.txt)",
        data=summary.encode("utf-8"),
        file_name=f"{stem}_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )

st.caption(
    "Recordings are handled temporarily during processing and are not intentionally stored by this app."
)
