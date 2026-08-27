from pathlib import Path
import os
import shutil
import tempfile

import streamlit as st

from cloud_transcriber import (
    TRANSCRIPTION_MODEL,
    TRANSLATION_MODEL,
    transcribe_chunks,
)
from media import split_to_audio_chunks
from summarizer import extractive_summary
from utils import transcript_to_srt, format_seconds

st.set_page_config(
    page_title="Jami Transcriber",
    page_icon="🎙️",
    layout="wide",
)

st.markdown("""
<style>
.block-container {max-width: 980px; padding-top: 2rem;}
[data-testid="stFileUploader"] {border-radius: 16px;}
.hero-sub {font-size: 1.02rem; opacity: .72; margin-top: -.4rem; margin-bottom: 1.1rem;}
.note {font-size: .9rem; opacity: .72;}
</style>
""", unsafe_allow_html=True)

st.title("🎙️ Jami Transcriber")
st.markdown(
    '<div class="hero-sub">Fast transcription for long recordings — optimized for Streamlit Community Cloud.</div>',
    unsafe_allow_html=True,
)

def configured_key() -> str:
    try:
        return str(st.secrets.get("GROQ_API_KEY", "") or "").strip()
    except Exception:
        return ""

secret_key = configured_key()

with st.sidebar:
    st.subheader("Settings")
    language_label = st.selectbox(
        "Language",
        ["Auto detect", "English", "Urdu"],
        index=0,
        help="Auto detect is best for mixed Urdu/English recordings.",
    )
    language_map = {"Auto detect": None, "English": "en", "Urdu": "ur"}

    output_label = st.selectbox(
        "Output",
        ["Keep spoken language", "Translate to English"],
        index=0,
    )
    translate = output_label == "Translate to English"

    timestamps = st.toggle("Timestamps in TXT", value=True)

    st.divider()
    st.caption(
        "This cloud version sends compressed audio chunks to Groq's speech-to-text API. "
        "The original uploaded file is not intentionally retained by this app."
    )

if secret_key:
    api_key = secret_key
    st.success("Cloud transcription engine is connected.", icon="✅")
else:
    st.info(
        "Enter a free Groq API key below. It is kept only in this Streamlit session "
        "unless you later add it to the app's Secrets."
    )
    api_key = st.text_input(
        "Groq API key",
        type="password",
        placeholder="gsk_...",
        help="Create a free key at console.groq.com/keys.",
    ).strip()

uploaded = st.file_uploader(
    "Upload audio or video",
    type=["mp3", "mp4", "m4a", "wav", "webm", "mpeg", "mpga", "ogg", "flac"],
    accept_multiple_files=False,
    help="MP3/M4A uploads are fastest. MP4 works too; only its audio track is transcribed.",
)

if uploaded is None:
    st.markdown(
        "**Workflow:** upload → compress to speech-only audio → transcribe in small chunks → "
        "merge timestamps → download TXT/SRT/summary."
    )
    st.stop()

suffix = Path(uploaded.name).suffix.lower() or ".media"
size_mb = uploaded.size / (1024 * 1024)

c1, c2, c3 = st.columns(3)
c1.metric("File", uploaded.name)
c2.metric("Upload", f"{size_mb:.1f} MB")
c3.metric("Mode", "Cloud Whisper")

if size_mb > 250:
    st.warning(
        "This is a large upload. It will work if Streamlit accepts it, but converting the source "
        "to MP3/M4A before uploading will be much faster."
    )

if not api_key:
    st.warning("Add the API key above to enable transcription.")
    st.stop()

if st.button("Transcribe recording", type="primary", use_container_width=True):
    st.session_state.pop("result", None)

    work_dir = Path(tempfile.mkdtemp(prefix="jami_transcriber_"))
    source_path = work_dir / f"source{suffix}"

    progress = st.progress(0, text="Saving upload…")
    status = st.status("Preparing recording", expanded=True)

    try:
        with source_path.open("wb") as out:
            uploaded.seek(0)
            shutil.copyfileobj(uploaded, out, length=1024 * 1024)

        progress.progress(4, text="Compressing speech audio…")
        status.write("Extracting the audio track and splitting it into lightweight chunks.")

        chunks = split_to_audio_chunks(
            input_path=str(source_path),
            output_dir=str(work_dir),
            segment_seconds=1200,
            bitrate="32k",
        )
        if not chunks:
            raise RuntimeError("No usable audio track was found in this file.")

        source_path.unlink(missing_ok=True)

        total_duration = sum(chunk["duration"] for chunk in chunks)
        status.write(
            f"Prepared {len(chunks)} chunk{'s' if len(chunks) != 1 else ''} "
            f"covering {format_seconds(total_duration)}."
        )

        model = TRANSLATION_MODEL if translate else TRANSCRIPTION_MODEL

        def api_progress(done: int, total: int, message: str):
            pct = 8 + int((done / max(1, total)) * 90)
            progress.progress(min(98, pct), text=message)
            status.write(message)

        result = transcribe_chunks(
            chunks=chunks,
            api_key=api_key,
            language=language_map[language_label],
            translate=translate,
            model=model,
            progress_callback=api_progress,
        )

        result["source_name"] = uploaded.name
        result["duration"] = total_duration
        st.session_state["result"] = result

        progress.progress(100, text="Transcription complete")
        status.update(label="Transcription complete", state="complete", expanded=False)

    except Exception as exc:
        status.update(label="Transcription failed", state="error", expanded=True)
        st.error(str(exc))
        st.info(
            "If this is an API/rate-limit message, wait briefly and retry. "
            "If it mentions the API key, check that the key begins with `gsk_`."
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

result = st.session_state.get("result")
if not result:
    st.stop()

segments = result.get("segments", [])
plain_text = result.get("text", "").strip()
duration = float(result.get("duration", 0.0))
source_name = result.get("source_name", uploaded.name)

st.success(
    f"Done · {format_seconds(duration)} · {len(segments)} timestamped segments"
)

if timestamps:
    txt_text = "\n".join(
        f"[{format_seconds(seg['start'])} → {format_seconds(seg['end'])}] {seg['text'].strip()}"
        for seg in segments if seg.get("text", "").strip()
    )
else:
    txt_text = plain_text

srt_text = transcript_to_srt(segments)
stem = Path(source_name).stem

tab1, tab2, tab3 = st.tabs(["Transcript", "Summary", "Downloads"])

with tab1:
    st.text_area("Transcript", txt_text, height=520, label_visibility="collapsed")

with tab2:
    summary_length = st.slider("Summary length", 3, 20, 8)
    summary = extractive_summary(plain_text, max_sentences=summary_length)
    if summary:
        st.markdown(summary)
        st.caption("This summary is generated locally from the transcript, with no extra AI API call.")
    else:
        st.info("Not enough transcript text to summarize.")

with tab3:
    summary = extractive_summary(plain_text, max_sentences=8)
    st.download_button(
        "Download transcript (.txt)",
        txt_text.encode("utf-8"),
        f"{stem}_transcript.txt",
        "text/plain",
        use_container_width=True,
    )
    st.download_button(
        "Download subtitles (.srt)",
        srt_text.encode("utf-8"),
        f"{stem}_transcript.srt",
        "application/x-subrip",
        use_container_width=True,
    )
    st.download_button(
        "Download summary (.txt)",
        summary.encode("utf-8"),
        f"{stem}_summary.txt",
        "text/plain",
        use_container_width=True,
    )

st.caption(
    "For confidential recordings, prefer running a local/offline transcription tool on your own computer."
)
