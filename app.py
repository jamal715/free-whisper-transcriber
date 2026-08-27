from pathlib import Path
import shutil
import tempfile

import streamlit as st

from cloud_transcriber import transcribe_chunks
from media import MAX_CHUNK_BYTES, probe_duration, split_audio_lossless
from summarizer import extractive_summary
from utils import transcript_to_srt, format_seconds

st.set_page_config(
    page_title="Jami Transcriber",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Jami Transcriber")
st.caption("Upload audio → transcribe → download text. That's it.")

def secret_key() -> str:
    try:
        return str(st.secrets.get("GROQ_API_KEY", "") or "").strip()
    except Exception:
        return ""

saved_key = secret_key()

if saved_key:
    api_key = saved_key
else:
    api_key = st.text_input(
        "Groq API key",
        type="password",
        placeholder="gsk_...",
        help="Your key is used only for the transcription request in this session.",
    ).strip()

language_label = st.selectbox(
    "Language",
    ["Auto detect", "English", "Urdu"],
    index=0,
)
language_map = {"Auto detect": None, "English": "en", "Urdu": "ur"}

uploaded = st.file_uploader(
    "Upload audio",
    type=["m4a", "mp3", "mp4", "wav", "flac", "ogg", "webm", "mpeg", "mpga"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Ready. Upload a recording above.")
    st.stop()

size_mb = uploaded.size / (1024 * 1024)
c1, c2 = st.columns(2)
c1.metric("File", uploaded.name)
c2.metric("Size", f"{size_mb:.1f} MB")

if uploaded.size > MAX_CHUNK_BYTES:
    st.caption(
        "Large file: it will be split into smaller pieces without re-encoding, "
        "then stitched back into one transcript."
    )
else:
    st.caption("This file can be sent directly to the transcription engine.")

if not api_key:
    st.warning("Enter your Groq API key first.")
    st.stop()

if st.button("Transcribe", type="primary", use_container_width=True):
    st.session_state.pop("result", None)
    work_dir = Path(tempfile.mkdtemp(prefix="jami_transcriber_"))
    suffix = Path(uploaded.name).suffix.lower() or ".m4a"
    source_path = work_dir / f"source{suffix}"

    progress = st.progress(0, text="Preparing file…")
    status = st.status("Transcribing", expanded=True)

    try:
        with source_path.open("wb") as out:
            uploaded.seek(0)
            shutil.copyfileobj(uploaded, out, length=1024 * 1024)

        progress.progress(5, text="Checking recording…")

        if source_path.stat().st_size <= MAX_CHUNK_BYTES:
            duration = probe_duration(str(source_path))
            chunks = [{
                "path": str(source_path),
                "duration": duration,
                "offset": 0.0,
            }]
            status.write("File fits in one request.")
        else:
            status.write("Splitting large recording without compressing or re-encoding it.")
            chunks = split_audio_lossless(
                input_path=str(source_path),
                output_dir=str(work_dir / "chunks"),
            )
            status.write(f"Prepared {len(chunks)} pieces.")

        total_duration = sum(float(c.get("duration", 0.0)) for c in chunks)

        def on_progress(done: int, total: int, message: str):
            pct = 10 + int((done / max(1, total)) * 88)
            progress.progress(min(98, pct), text=message)
            status.write(message)

        result = transcribe_chunks(
            chunks=chunks,
            api_key=api_key,
            language=language_map[language_label],
            progress_callback=on_progress,
        )

        result["duration"] = total_duration or result.get("duration", 0.0)
        result["source_name"] = uploaded.name
        st.session_state["result"] = result

        progress.progress(100, text="Done")
        status.update(label="Transcription complete", state="complete", expanded=False)

    except Exception as exc:
        status.update(label="Transcription failed", state="error", expanded=True)
        st.error(str(exc))
        st.caption(
            "For best reliability, use M4A or MP3. Large files are split without audio re-encoding."
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

st.success(f"Done · {format_seconds(duration)}")

timestamped = "\n".join(
    f"[{format_seconds(seg['start'])} → {format_seconds(seg['end'])}] {seg['text'].strip()}"
    for seg in segments
    if seg.get("text", "").strip()
)

tab1, tab2, tab3 = st.tabs(["Transcript", "Summary", "Download"])

with tab1:
    st.text_area(
        "Transcript",
        value=timestamped or plain_text,
        height=500,
        label_visibility="collapsed",
    )

with tab2:
    summary = extractive_summary(plain_text, max_sentences=8)
    if summary:
        st.markdown(summary)
    else:
        st.info("Not enough text to summarize.")

with tab3:
    stem = Path(source_name).stem
    srt_text = transcript_to_srt(segments)
    st.download_button(
        "Download transcript (.txt)",
        data=(timestamped or plain_text).encode("utf-8"),
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
