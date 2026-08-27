from pathlib import Path

import streamlit as st

from cloud_transcriber import transcribe_upload
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
</style>
""", unsafe_allow_html=True)

st.title("🎙️ Jami Transcriber")
st.markdown(
    '<div class="hero-sub">Upload → Groq Whisper → transcript. No local audio conversion or AI inference.</div>',
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
        "CPU-light mode: the app does not run Whisper or FFmpeg. "
        "The uploaded recording is sent to Groq for speech-to-text."
    )

if secret_key:
    api_key = secret_key
    st.success("Cloud transcription engine is connected.", icon="✅")
else:
    st.info(
        "Enter your Groq API key. It is kept only in this Streamlit session "
        "unless you add it to the app's Secrets."
    )
    api_key = st.text_input(
        "Groq API key",
        type="password",
        placeholder="gsk_...",
        help="Create a key at console.groq.com/keys.",
    ).strip()

uploaded = st.file_uploader(
    "Upload audio or video",
    type=["mp3", "mp4", "m4a", "wav", "webm", "mpeg", "mpga", "ogg", "flac"],
    accept_multiple_files=False,
    help="The file is sent as-is to Groq. MP3/M4A are usually the fastest uploads.",
)

if uploaded is None:
    st.markdown(
        "**Workflow:** upload → send directly to Groq Whisper → receive timestamped transcript → "
        "download TXT/SRT/summary."
    )
    st.stop()

size_mb = uploaded.size / (1024 * 1024)

c1, c2, c3 = st.columns(3)
c1.metric("File", uploaded.name)
c2.metric("Upload", f"{size_mb:.1f} MB")
c3.metric("Processing", "Groq Cloud")

if size_mb <= 24:
    st.caption("This file will use Groq's normal direct-upload path.")
else:
    st.caption(
        "This file is above Groq's free-tier attachment size, so the app will use "
        "Groq's larger-file URL/Base64URL path. No FFmpeg compression is performed."
    )

if size_mb > 100:
    st.warning(
        "Files above 100 MB can use a lot of browser/server memory even without audio processing. "
        "For very large videos, an audio-only M4A/MP3 copy is recommended."
    )

if not api_key:
    st.warning("Add the Groq API key above to enable transcription.")
    st.stop()

if st.button("Transcribe recording", type="primary", use_container_width=True):
    st.session_state.pop("result", None)
    progress = st.progress(5, text="Sending recording to Groq…")
    status = st.status("Cloud transcription", expanded=True)

    try:
        status.write("No local conversion is running. Your Streamlit CPU stays almost idle.")
        progress.progress(20, text="Uploading to Groq Whisper…")

        result = transcribe_upload(
            uploaded_file=uploaded,
            filename=uploaded.name,
            size_bytes=uploaded.size,
            api_key=api_key,
            language=language_map[language_label],
            translate=translate,
            content_type=getattr(uploaded, "type", None),
        )
        result["source_name"] = uploaded.name
        st.session_state["result"] = result

        progress.progress(100, text="Transcription complete")
        status.update(label="Transcription complete", state="complete", expanded=False)

    except Exception as exc:
        status.update(label="Transcription failed", state="error", expanded=True)
        st.error(str(exc))
        st.info(
            "If Groq reports a size or URL error for a very large file, convert it once to M4A/MP3 "
            "or use a public file URL. API/rate-limit errors can usually be retried after a short wait."
        )

result = st.session_state.get("result")
if not result:
    st.stop()

segments = result.get("segments", [])
plain_text = result.get("text", "").strip()
duration = float(result.get("duration", 0.0))
source_name = result.get("source_name", uploaded.name)
transfer_mode = result.get("transfer_mode", "cloud")

st.success(
    f"Done · {format_seconds(duration)} · {len(segments)} timestamped segments · {transfer_mode}"
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
        st.caption("The quick summary is generated locally from text only; no audio processing is done here.")
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
    "Privacy note: this cloud version sends the recording to Groq for transcription. "
    "For confidential recordings, use an offline/local transcriber."
)
