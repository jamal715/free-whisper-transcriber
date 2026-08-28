from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import shutil
import tempfile

import streamlit as st

from cloud_transcriber import (
    ACCURACY_MODEL,
    FAST_MODEL,
    transcribe_chunks,
    translate_chunks,
)
from media import MAX_CHUNK_BYTES, probe_duration, split_audio_lossless
from summarizer import extractive_summary
from utils import transcript_to_srt, format_seconds

st.set_page_config(
    page_title="Jami Research Transcriber",
    page_icon="🎙️",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {max-width: 1180px; padding-top: 1.5rem;}
.hero-note {opacity: .72; margin-top: -.4rem; margin-bottom: 1rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🎙️ Jami Research Transcriber")
st.markdown(
    '<div class="hero-note">Long-form interview transcription for Urdu + English research audio.</div>',
    unsafe_allow_html=True,
)


def configured_key() -> str:
    try:
        return str(st.secrets.get("GROQ_API_KEY", "") or "").strip()
    except Exception:
        return ""


def segments_to_csv(segments: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["start_seconds", "end_seconds", "start", "end", "text", "review_flag"])
    for seg in segments:
        writer.writerow([
            f"{float(seg.get('start', 0.0)):.3f}",
            f"{float(seg.get('end', 0.0)):.3f}",
            format_seconds(float(seg.get("start", 0.0))),
            format_seconds(float(seg.get("end", 0.0))),
            str(seg.get("text", "")).strip(),
            "YES" if seg.get("review_flag") else "",
        ])
    return buf.getvalue()


def timestamped_text(segments: list[dict]) -> str:
    return "\n".join(
        f"[{format_seconds(float(seg.get('start', 0.0)))} → "
        f"{format_seconds(float(seg.get('end', 0.0)))}] "
        f"{str(seg.get('text', '')).strip()}"
        for seg in segments
        if str(seg.get("text", "")).strip()
    )


saved_key = configured_key()

with st.sidebar:
    st.subheader("Research settings")

    mode_label = st.radio(
        "Accuracy",
        ["Research accuracy", "Fast draft"],
        index=0,
        help=(
            "Research accuracy uses Whisper Large V3. Fast draft uses "
            "Whisper Large V3 Turbo."
        ),
    )
    model = ACCURACY_MODEL if mode_label == "Research accuracy" else FAST_MODEL

    language_label = st.selectbox(
        "Interview language",
        ["Auto detect — recommended", "Urdu", "English"],
        index=0,
        help="Keep Auto detect for interviews that naturally switch between Urdu and English.",
    )
    language_map = {
        "Auto detect — recommended": None,
        "Urdu": "ur",
        "English": "en",
    }

    make_translation = st.checkbox(
        "Also create English translation",
        value=False,
        help="Creates a separate English rendering after preserving the original transcript.",
    )

    context = st.text_area(
        "Names / places / research terms (optional)",
        placeholder="e.g. Layyah, Kot Sultan, school names, project terms...",
        height=100,
        max_chars=900,
        help=(
            "Whisper can use a short context prompt to improve unusual names and spellings. "
            "Do not put confidential identifiers here unless needed."
        ),
    )

    st.divider()
    st.caption(
        "Original transcript is always produced first. Translation is kept separate "
        "so the source-language record is never overwritten."
    )

if saved_key:
    api_key = saved_key
    st.success("Transcription service connected.", icon="✅")
else:
    api_key = st.text_input(
        "Groq API key",
        type="password",
        placeholder="gsk_...",
        help="For the permanent app, store this once in Streamlit Secrets instead.",
    ).strip()
    st.caption("The key is not written into the GitHub repository.")

uploaded = st.file_uploader(
    "Upload interview audio or video",
    type=["m4a", "mp3", "mp4", "wav", "flac", "ogg", "webm", "mpeg", "mpga"],
    accept_multiple_files=False,
    help="M4A or MP3 is recommended for 60–90 minute interviews.",
)

if uploaded is None:
    st.info("Upload an interview when ready. Nothing is processed until you press Transcribe.")
    with st.expander("What this version is designed for"):
        st.markdown(
            "- 60–90+ minute research interviews\n"
            "- Urdu / English code-switching\n"
            "- accuracy-first or fast-draft mode\n"
            "- timestamped transcript + SRT + CSV + JSON\n"
            "- optional separate English translation\n"
            "- automatic review flags for uncertain-looking segments"
        )
    st.stop()

size_mb = uploaded.size / (1024 * 1024)

m1, m2, m3 = st.columns(3)
m1.metric("File", uploaded.name)
m2.metric("Size", f"{size_mb:.1f} MB")
m3.metric("Model", "Large V3" if model == ACCURACY_MODEL else "Large V3 Turbo")

if uploaded.size > MAX_CHUNK_BYTES:
    st.caption(
        "Large recording: it will be stream-split into smaller pieces without re-encoding, "
        "then reassembled into one continuous transcript."
    )
else:
    st.caption("This recording fits in a single transcription request.")

if not api_key:
    st.warning("Enter the Groq API key before starting.")
    st.stop()

if st.button("Transcribe interview", type="primary", use_container_width=True):
    st.session_state.pop("result", None)
    work_dir = Path(tempfile.mkdtemp(prefix="jami_research_"))
    suffix = Path(uploaded.name).suffix.lower() or ".m4a"
    source_path = work_dir / f"source{suffix}"
    chunk_dir = work_dir / "chunks"

    progress = st.progress(0, text="Preparing interview…")
    status = st.status("Research transcription", expanded=True)

    try:
        with source_path.open("wb") as out:
            uploaded.seek(0)
            shutil.copyfileobj(uploaded, out, length=1024 * 1024)

        duration = probe_duration(str(source_path))
        if duration:
            status.write(f"Recording length: {format_seconds(duration)}.")

        if source_path.stat().st_size <= MAX_CHUNK_BYTES:
            chunks = [{
                "path": str(source_path),
                "duration": duration,
                "offset": 0.0,
            }]
        else:
            progress.progress(5, text="Splitting large recording safely…")
            status.write("Splitting without re-encoding the audio.")
            chunks = split_audio_lossless(str(source_path), str(chunk_dir))

        status.write(f"Prepared {len(chunks)} part{'s' if len(chunks) != 1 else ''}.")

        def original_progress(done: int, total: int, message: str):
            pct = 8 + int((done / max(1, total)) * (54 if make_translation else 88))
            progress.progress(min(94, pct), text=message)
            status.write(message)

        result = transcribe_chunks(
            chunks=chunks,
            api_key=api_key,
            language=language_map[language_label],
            model=model,
            context_prompt=context.strip(),
            progress_callback=original_progress,
        )

        if make_translation:
            status.write("Original transcript complete. Creating separate English translation…")

            def translation_progress(done: int, total: int, message: str):
                pct = 62 + int((done / max(1, total)) * 34)
                progress.progress(min(96, pct), text=message)

            translation = translate_chunks(
                chunks=chunks,
                api_key=api_key,
                context_prompt=context.strip(),
                progress_callback=translation_progress,
            )
            result["translation"] = translation

        result["source_name"] = uploaded.name
        result["requested_model"] = model
        st.session_state["result"] = result

        progress.progress(100, text="Interview ready")
        status.update(label="Transcription complete", state="complete", expanded=False)

    except Exception as exc:
        status.update(label="Transcription failed", state="error", expanded=True)
        st.error(str(exc))
        st.info(
            "The app retries temporary network/rate-limit failures automatically. "
            "If a failure persists, keep the original recording and retry later."
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

result = st.session_state.get("result")
if not result:
    st.stop()

segments = result.get("segments", [])
plain_text = str(result.get("text", "") or "").strip()
duration = float(result.get("duration", 0.0) or 0.0)
source_name = result.get("source_name", uploaded.name)
review_segments = [seg for seg in segments if seg.get("review_flag")]
detected_languages = result.get("languages", [])
model_used = result.get("model", result.get("requested_model", "unknown"))

st.success("Interview transcription is ready.")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Duration", format_seconds(duration))
r2.metric("Parts", int(result.get("parts", 1)))
r3.metric("Segments", len(segments))
r4.metric("Review flags", len(review_segments))

if detected_languages:
    st.caption("Detected language by part: " + " · ".join(detected_languages))

tab_names = ["Original transcript"]
if result.get("translation"):
    tab_names.append("English translation")
tab_names += ["Review flags", "Summary", "Downloads"]

tabs = st.tabs(tab_names)
idx = 0

with tabs[idx]:
    idx += 1
    original_timestamped = timestamped_text(segments)
    st.text_area(
        "Timestamped original",
        value=original_timestamped,
        height=600,
        help="Use this as the source record; it preserves the original transcription.",
    )

if result.get("translation"):
    with tabs[idx]:
        idx += 1
        translated = result["translation"]
        translation_segments = translated.get("segments", [])
        translated_text = timestamped_text(translation_segments) or translated.get("text", "")
        st.text_area(
            "English translation",
            value=translated_text,
            height=600,
            help="This is a separate translation; the original transcript remains unchanged.",
        )

with tabs[idx]:
    idx += 1
    if review_segments:
        st.warning(
            "These are heuristic review flags, not proof of an error. "
            "Listen back to these timestamps before quoting them in research."
        )
        rows = [{
            "time": f"{format_seconds(seg['start'])}–{format_seconds(seg['end'])}",
            "text": seg["text"],
            "reason": ", ".join(seg.get("review_reasons", [])),
        } for seg in review_segments]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.success("No segments crossed the automatic review thresholds.")

with tabs[idx]:
    idx += 1
    summary_length = st.slider("Summary length", 4, 20, 10)
    summary = extractive_summary(plain_text, max_sentences=summary_length)
    if summary:
        st.markdown(summary)
        st.caption("This is an extractive convenience summary, not a substitute for qualitative coding.")
    else:
        st.info("Not enough text to summarize.")

with tabs[idx]:
    stem = Path(source_name).stem
    original_timestamped = timestamped_text(segments)
    original_srt = transcript_to_srt(segments)
    original_csv = segments_to_csv(segments)

    export_json = {
        "source_file": source_name,
        "duration_seconds": duration,
        "model": model_used,
        "languages": detected_languages,
        "text": plain_text,
        "segments": segments,
    }
    if result.get("translation"):
        export_json["translation"] = result["translation"]

    st.download_button(
        "Download original transcript (.txt)",
        original_timestamped.encode("utf-8"),
        f"{stem}_original_transcript.txt",
        "text/plain",
        use_container_width=True,
    )
    st.download_button(
        "Download subtitles (.srt)",
        original_srt.encode("utf-8"),
        f"{stem}_original.srt",
        "application/x-subrip",
        use_container_width=True,
    )
    st.download_button(
        "Download research table (.csv)",
        original_csv.encode("utf-8-sig"),
        f"{stem}_segments.csv",
        "text/csv",
        use_container_width=True,
    )
    st.download_button(
        "Download full metadata (.json)",
        json.dumps(export_json, ensure_ascii=False, indent=2).encode("utf-8"),
        f"{stem}_transcription.json",
        "application/json",
        use_container_width=True,
    )

    if result.get("translation"):
        translated = result["translation"]
        translated_txt = timestamped_text(translated.get("segments", [])) or translated.get("text", "")
        st.download_button(
            "Download English translation (.txt)",
            translated_txt.encode("utf-8"),
            f"{stem}_english_translation.txt",
            "text/plain",
            use_container_width=True,
        )

st.caption(
    "Privacy: temporary working files are deleted after processing. "
    "For sensitive interviews, enable Zero Data Retention in Groq Data Controls "
    "or use the planned offline/local engine."
)
