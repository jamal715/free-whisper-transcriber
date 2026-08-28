from __future__ import annotations

import csv
import html
import io
import json
from pathlib import Path
import shutil
import tempfile

import streamlit as st

from brand import LOGO_DATA_URI
from cloud_transcriber import ACCURACY_MODEL, FAST_MODEL, transcribe_chunks
from media import MAX_CHUNK_BYTES, probe_duration, split_audio_lossless
from openai_validator import transcribe_for_validation
from quality import assess_dual, overall_health
from text_postprocess import translate_validated_chunks, build_research_summary
from utils import transcript_to_srt, format_seconds

PRIMARY = "#004d73"
MAX_DURATION_SECONDS = 120 * 60

st.set_page_config(
    page_title="QuantOra Research Transcriber",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
<style>
:root {{
  --q: {PRIMARY};
  --ink: #111418;
  --muted: #69737d;
  --line: #dce2e6;
  --soft: #f6f8f9;
  --blue-soft: #f4f9fb;
}}
[data-testid="stSidebar"] {{display:none;}}
[data-testid="stHeader"] {{background:rgba(255,255,255,.96);}}
.stApp {{background:#fff; color:var(--ink);}}
.block-container {{max-width:1480px; padding:1.1rem 2.2rem 3.2rem;}}
h1,h2,h3,h4,p,label,div {{font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}}
.q-top {{
  display:flex; align-items:center; justify-content:space-between;
  border-bottom:1px solid var(--line); padding:2px 2px 18px; margin-bottom:28px;
}}
.q-brand {{display:flex; align-items:center; gap:14px; min-width:0;}}
.q-brand img {{
  width:62px; height:62px; object-fit:cover; border-radius:16px;
  box-shadow:0 1px 5px rgba(0,0,0,.09);
}}
.q-name {{font-size:29px; font-weight:800; letter-spacing:-.8px; color:#111418;}}
.q-divider {{width:1px;height:38px;background:#dce2e6;margin:0 5px;}}
.q-tag {{font-size:14px;color:#58636e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:510px;}}
.q-secure-pill {{
  border:1px solid #cdd8de; border-radius:10px; color:var(--q); background:#fbfcfd;
  padding:9px 13px; font-size:13px; font-weight:700;
}}
.q-hero {{text-align:center; padding:10px 0 22px;}}
.q-hero h1 {{font-size:42px; line-height:1.05; margin:.15rem 0 .6rem; letter-spacing:-1.5px;}}
.q-hero p {{margin:0; color:#68727e; font-size:17px;}}
.q-side-title {{font-size:22px;font-weight:800;margin:2px 0 12px;}}
.q-note {{
  border:1px solid var(--line); border-radius:12px; padding:14px 15px;
  background:#fafcfd; color:#606b75; font-size:12px; line-height:1.55;
}}
.q-connected {{color:var(--q);font-size:12px;font-weight:700;margin-top:7px;}}
.q-health {{
  border-left:4px solid var(--q); background:#f5fafc; padding:14px 16px;
  border-radius:9px; margin:8px 0 16px;
}}
.q-health strong {{font-size:16px;}}
.q-chunk {{border-bottom:1px solid #e7ebee;padding:13px 2px 15px;}}
.q-time {{font-size:12px;color:var(--q);font-weight:750;}}
.q-text {{font-size:15px;line-height:1.72;color:#171a1d;margin-top:5px;}}
.q-provider {{font-size:11px;color:#8b949b;margin-top:5px;}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
  border-color:var(--line)!important; border-radius:14px!important;
  box-shadow:0 1px 2px rgba(0,0,0,.025);
}}
div[data-testid="stFileUploader"] section {{
  min-height:215px; border:1.5px dashed #a9c6d4; background:#fbfdfe; border-radius:14px;
}}
div[data-testid="stFileUploader"] button {{
  background:var(--q)!important;color:#fff!important;border:none!important;border-radius:8px!important;
}}
.stButton > button[kind="primary"] {{
  background:var(--q)!important;color:white!important;border:1px solid var(--q)!important;
  min-height:49px;font-weight:750;border-radius:9px;
}}
.stButton > button:disabled {{opacity:.58!important;}}
.stTabs [data-baseweb="tab-list"] {{gap:24px;border-bottom:1px solid var(--line);}}
.stTabs [aria-selected="true"] {{color:var(--q)!important;}}
div[data-testid="stMetricValue"] {{font-size:1.42rem;color:#111820;}}
div[data-testid="stAlert"] {{border-radius:10px;}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="q-top">
  <div class="q-brand">
    <img src="{LOGO_DATA_URI}" alt="QuantOra logo">
    <div class="q-name">QuantOra</div>
    <div class="q-divider"></div>
    <div class="q-tag">Quantifying economic signals for a sustainable future.</div>
  </div>
  <div class="q-secure-pill">🔒 Research workspace</div>
</div>
<div class="q-hero">
  <h1>QuantOra Research Transcriber</h1>
  <p>High-confidence transcription, validation and English translation for Urdu–English research interviews.</p>
</div>
""",
    unsafe_allow_html=True,
)


def secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "") or "").strip()
    except Exception:
        return ""


def timestamped_text(segments: list[dict]) -> str:
    return "\n".join(
        f"[{format_seconds(float(seg.get('start', 0.0)))} → {format_seconds(float(seg.get('end', 0.0)))}] "
        f"{str(seg.get('text', '')).strip()}"
        for seg in segments
        if str(seg.get("text", "")).strip()
    )


def validated_text(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{format_seconds(float(c.get('start', 0.0)))} → {format_seconds(float(c.get('end', 0.0)))}]\n"
        f"{str(c.get('selected_text', '')).strip()}"
        for c in chunks
        if str(c.get("selected_text", "")).strip()
    )


def translation_text(translation: dict) -> str:
    return "\n\n".join(
        f"[{format_seconds(float(c.get('start', 0.0)))} → {format_seconds(float(c.get('end', 0.0)))}]\n"
        f"{str(c.get('text', '')).strip()}"
        for c in translation.get("chunks", [])
        if str(c.get("text", "")).strip()
    )


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


saved_groq = secret("GROQ_API_KEY")
saved_openai = secret("OPENAI_API_KEY")

left, main = st.columns([0.24, 0.76], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="q-side-title">Language</div>', unsafe_allow_html=True)
        language_label = st.selectbox(
            "Language",
            ["Urdu + English (mixed)", "Mostly Urdu", "Mostly English"],
            index=0,
            label_visibility="collapsed",
            help="For natural Pakistani Urdu-English code-switching, keep Mixed.",
        )
        language_map = {
            "Urdu + English (mixed)": None,
            "Mostly Urdu": "ur",
            "Mostly English": "en",
        }

        st.markdown('<div class="q-side-title" style="margin-top:20px;">Transcription mode</div>', unsafe_allow_html=True)
        mode_label = st.selectbox(
            "Mode",
            ["Maximum confidence", "Research accuracy", "Fast draft"],
            index=0,
            label_visibility="collapsed",
            help="Maximum confidence uses Groq Whisper Large V3 plus independent OpenAI validation when configured.",
        )
        model = FAST_MODEL if mode_label == "Fast draft" else ACCURACY_MODEL

        st.markdown('<div class="q-side-title" style="margin-top:20px;">Translation</div>', unsafe_allow_html=True)
        make_translation = st.toggle(
            "Create English translation",
            value=True,
            help="Translation is generated from the validated transcript, not by resending the full audio.",
        )

        with st.expander("Advanced"):
            context = st.text_area(
                "Names / places / research terms",
                placeholder="HBL, PASCO, IFC, Layyah, company names, technical terms…",
                height=95,
                max_chars=900,
            )
            groq_key = saved_groq or st.text_input(
                "Groq API key",
                type="password",
                placeholder="gsk_…",
                key="groq_key_input",
            ).strip()
            openai_key = saved_openai or st.text_input(
                "OpenAI API key (optional)",
                type="password",
                placeholder="sk-…",
                key="openai_key_input",
                help="Used only for independent validation/rescue in Maximum confidence mode.",
            ).strip()

            if saved_groq:
                st.markdown('<div class="q-connected">● Groq connected from Secrets</div>', unsafe_allow_html=True)
            if saved_openai:
                st.markdown('<div class="q-connected">● OpenAI validator connected from Secrets</div>', unsafe_allow_html=True)

    st.markdown(
        """
<div class="q-note">
<strong>Privacy by design</strong><br>
Temporary working files are deleted after each run. Keep API keys in Streamlit Secrets.
For sensitive research data, use provider Zero Data Retention settings where available.
</div>
""",
        unsafe_allow_html=True,
    )

with main:
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Drag & drop your audio or video file here",
            type=["m4a", "mp3", "mp4", "wav", "flac", "ogg", "webm", "mpeg", "mpga"],
            accept_multiple_files=False,
            help="Designed for interviews up to 120 minutes. MP3/M4A is recommended for long interviews.",
        )

    ready = uploaded is not None and bool(groq_key)

    if uploaded is not None:
        size_mb = uploaded.size / (1024 * 1024)
        c1, c2, c3 = st.columns(3)
        c1.metric("File", uploaded.name)
        c2.metric("Size", f"{size_mb:.1f} MB")
        if mode_label == "Maximum confidence" and openai_key:
            c3.metric("Validation", "Dual engine")
        elif mode_label == "Maximum confidence":
            c3.metric("Validation", "Integrity scan")
        else:
            c3.metric("Model", "Large V3" if model == ACCURACY_MODEL else "Large V3 Turbo")

        if uploaded.size > MAX_CHUNK_BYTES:
            st.caption(
                "Large recording detected. It will be split into safe chunks automatically; "
                "overlap is adjusted near two hours to stay inside the speech quota."
            )

    if not groq_key:
        st.info("Add your Groq key under Advanced to enable transcription.")

    if mode_label == "Maximum confidence" and not openai_key:
        st.warning(
            "Maximum confidence is currently using one speech engine plus corruption checks. "
            "Add an OpenAI API key under Advanced for independent chunk-by-chunk validation and rescue."
        )

    action_label = "Transcribe + validate"
    if make_translation:
        action_label += " + translate"

    start = st.button(
        action_label if ready else "Waiting for file and API key…",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    )

    if start:
        st.session_state.pop("result", None)
        st.session_state.pop("research_summary", None)
        work_dir = Path(tempfile.mkdtemp(prefix="quantora_research_"))
        suffix = Path(uploaded.name).suffix.lower() or ".m4a"
        source_path = work_dir / f"source{suffix}"
        chunk_dir = work_dir / "chunks"

        progress = st.progress(0, text="Receiving interview…")
        status = st.status("QuantOra processing pipeline", expanded=True)

        try:
            with source_path.open("wb") as out:
                uploaded.seek(0)
                shutil.copyfileobj(uploaded, out, length=1024 * 1024)

            duration = probe_duration(str(source_path))
            if not duration:
                raise RuntimeError("Could not read the recording duration. For long interviews, use MP3 or M4A.")
            status.write(f"Recording length: {format_seconds(duration)}")

            if duration > MAX_DURATION_SECONDS + 0.5:
                raise RuntimeError(
                    f"This workspace is configured for a maximum of 120 minutes. "
                    f"Your recording is {format_seconds(duration)}."
                )

            if source_path.stat().st_size <= MAX_CHUNK_BYTES:
                chunks = [{
                    "path": str(source_path),
                    "duration": duration,
                    "offset": 0.0,
                    "keep_after": 0.0,
                    "nominal_end": duration,
                    "overlap_seconds": 0.0,
                }]
            else:
                progress.progress(4, text="Preparing safe chunks…")
                status.write("Splitting the recording without re-encoding.")
                chunks = split_audio_lossless(str(source_path), str(chunk_dir))

            total_submitted = sum(float(c.get("duration", 0.0)) for c in chunks)
            overlap_total = max(0.0, total_submitted - duration)
            status.write(
                f"Prepared {len(chunks)} part{'s' if len(chunks) != 1 else ''}; "
                f"boundary overlap: {overlap_total:.1f}s total."
            )

            mixed_prompt = ""
            if language_label == "Urdu + English (mixed)":
                mixed_prompt = (
                    "Pakistani research interview with natural Urdu-English code-switching. "
                    "Write Urdu speech in Urdu Perso-Arabic script and English speech in Latin script. "
                    "Do not output Devanagari/Hindi script. Preserve numbers, names, acronyms and technical terms exactly. "
                )
            speech_context = (mixed_prompt + context.strip())[:850]

            def transcription_progress(done: int, total: int, message: str):
                ceiling = 46 if (mode_label == "Maximum confidence" and openai_key) else 70
                pct = 6 + int((done / max(1, total)) * ceiling)
                progress.progress(min(78, pct), text=message)
                status.write(message)

            result = transcribe_chunks(
                chunks=chunks,
                api_key=groq_key,
                language=language_map[language_label],
                model=model,
                context_prompt=speech_context,
                progress_callback=transcription_progress,
            )

            validated_chunks = []
            verifier_model = None
            chunk_results = result.get("chunk_results", [])

            for idx, chunk_result in enumerate(chunk_results):
                verifier_text = None
                verifier_error = None

                if mode_label == "Maximum confidence" and openai_key:
                    progress.progress(
                        54 + int((idx / max(1, len(chunk_results))) * 25),
                        text=f"Independent validation: part {idx + 1} of {len(chunk_results)}…",
                    )
                    status.write(f"Independent validator is checking part {idx + 1}.")
                    try:
                        verification = transcribe_for_validation(
                            path=chunks[idx]["path"],
                            api_key=openai_key,
                            context_prompt=(
                                "Pakistani research interview. Speech may naturally switch between Urdu and English. "
                                "Preserve the spoken language and wording. Urdu should use Urdu Perso-Arabic script, "
                                "English should remain Latin script. Never output Devanagari/Hindi unless it is genuinely spoken. "
                                "Preserve names, numbers, acronyms and technical terms. Do not translate or invent. "
                                + context.strip()
                            )[:900],
                        )
                        verifier_text = verification["text"]
                        verifier_model = verification.get("model")
                    except Exception as exc:
                        verifier_error = str(exc)

                assessment = assess_dual(chunk_result.get("text", ""), verifier_text)
                assessment.update({
                    "index": idx + 1,
                    "start": float(chunk_result.get("keep_after", chunk_result.get("offset", 0.0))),
                    "end": float(chunk_result.get("end", 0.0)),
                    "groq_text": chunk_result.get("text", ""),
                    "verifier_text": verifier_text,
                    "verifier_error": verifier_error,
                })

                if verifier_error:
                    assessment["reasons"] = list(assessment.get("reasons", [])) + [
                        "independent validator unavailable for this chunk"
                    ]
                    if assessment["status"] == "passed":
                        assessment["status"] = "review"
                        assessment["score"] = min(int(assessment["score"]), 78)

                validated_chunks.append(assessment)

            health = overall_health(validated_chunks)
            result["validated_chunks"] = validated_chunks
            result["health"] = health
            result["verifier_model"] = verifier_model
            result["validated_text"] = "\n\n".join(
                c.get("selected_text", "").strip()
                for c in validated_chunks
                if c.get("selected_text", "").strip()
            )

            if make_translation:
                status.write("Validation complete. Translating the validated transcript to English.")

                def translation_progress(done: int, total: int, message: str):
                    pct = 80 + int((done / max(1, total)) * 17)
                    progress.progress(min(97, pct), text=message)
                    status.write(message)

                result["translation"] = translate_validated_chunks(
                    validated_chunks=validated_chunks,
                    api_key=groq_key,
                    progress_callback=translation_progress,
                )

            result["source_name"] = uploaded.name
            result["requested_model"] = model
            st.session_state["result"] = result

            progress.progress(100, text="Interview ready")
            if health.get("failed", 0):
                status.update(label="Completed — failed sections require review", state="complete", expanded=False)
            elif health.get("review", 0):
                status.update(label="Completed — review flagged sections", state="complete", expanded=False)
            else:
                status.update(label="Completed — high confidence", state="complete", expanded=False)

        except Exception as exc:
            status.update(label="Processing failed", state="error", expanded=True)
            st.error(str(exc))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


result = st.session_state.get("result")
if result:
    validated_chunks = result.get("validated_chunks", [])
    health = result.get("health", {})
    segments = result.get("segments", [])
    duration = float(result.get("duration", 0.0) or 0.0)
    source_name = result.get("source_name", "interview")

    st.markdown("---")

    status_text = str(health.get("status", "unknown")).title()
    st.markdown(
        f"""
<div class="q-health">
  <strong>Transcript health: {html.escape(status_text)}</strong>
  &nbsp;·&nbsp; Score {health.get('score', 0)}/100
  &nbsp;·&nbsp; {health.get('passed', 0)} passed
  &nbsp;·&nbsp; {health.get('review', 0)} review
  &nbsp;·&nbsp; {health.get('failed', 0)} failed
</div>
""",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration", format_seconds(duration))
    m2.metric("Parts", int(result.get("parts", 1)))
    m3.metric("Reliability", f"{health.get('score', 0)}/100")
    m4.metric("Validation", "Dual engine" if result.get("verifier_model") else "Single engine + QA")

    tabs = st.tabs(["Transcript", "Translation", "Validation", "Summary", "Downloads"])

    with tabs[0]:
        for chunk in validated_chunks:
            state = chunk.get("status", "review")
            icon = "✓" if state == "passed" else "!" if state == "review" else "×"
            provider = str(chunk.get("selected_provider", "Groq"))
            safe = html.escape(str(chunk.get("selected_text", ""))).replace("\n", "<br>")
            st.markdown(
                f"""
<div class="q-chunk">
  <div class="q-time">{format_seconds(float(chunk.get('start', 0.0)))} — {format_seconds(float(chunk.get('end', 0.0)))} &nbsp; {icon} {state.upper()} · {chunk.get('score', 0)}/100</div>
  <div class="q-text">{safe}</div>
  <div class="q-provider">{html.escape(provider)}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        with st.expander("Raw detailed timestamps"):
            st.text_area(
                "Raw segment transcript",
                value=timestamped_text(segments),
                height=520,
                label_visibility="collapsed",
            )

    with tabs[1]:
        if result.get("translation"):
            st.text_area(
                "English translation",
                value=translation_text(result["translation"]),
                height=650,
                label_visibility="collapsed",
            )
            st.caption(
                "This English version is translated from the validated transcript, so it does not consume another 120 minutes of speech quota."
            )
        else:
            st.info("Translation was not selected for this run.")

    with tabs[2]:
        rows = []
        for chunk in validated_chunks:
            similarity = chunk.get("similarity")
            rows.append({
                "time": f"{format_seconds(float(chunk.get('start', 0.0)))}–{format_seconds(float(chunk.get('end', 0.0)))}",
                "status": chunk.get("status"),
                "score": chunk.get("score"),
                "engine agreement": "" if similarity is None else f"{float(similarity):.0%}",
                "selected": chunk.get("selected_provider"),
                "reason": ", ".join(chunk.get("reasons", [])),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            "Use this panel to decide where you actually need to listen back. Direct quotations for publication should still be checked against the audio."
        )

    with tabs[3]:
        if "research_summary" not in st.session_state:
            st.caption("Generate a clean English summary plus a Roman Urdu version from the validated transcript.")
            if st.button("Generate research summary", use_container_width=True, key="generate_summary"):
                with st.spinner("Building research summary…"):
                    try:
                        st.session_state["research_summary"] = build_research_summary(
                            validated_chunks=validated_chunks,
                            api_key=groq_key,
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
        else:
            summary = st.session_state["research_summary"]
            st.markdown("### English summary")
            st.markdown(summary.get("english", "") or "_No summary returned._")
            st.markdown("### Roman Urdu summary")
            st.markdown(summary.get("roman_urdu", "") or "_No Roman Urdu summary returned._")
            st.caption(f"Summary model: {summary.get('model', 'unknown')}")

    with tabs[4]:
        stem = Path(source_name).stem
        validation_json = {
            "source_file": source_name,
            "health": health,
            "model": result.get("model"),
            "verifier_model": result.get("verifier_model"),
            "validated_chunks": validated_chunks,
        }

        st.download_button(
            "Download validated transcript (.txt)",
            validated_text(validated_chunks).encode("utf-8"),
            f"{stem}_validated_transcript.txt",
            "text/plain",
            use_container_width=True,
        )

        if result.get("translation"):
            st.download_button(
                "Download English translation (.txt)",
                translation_text(result["translation"]).encode("utf-8"),
                f"{stem}_english_translation.txt",
                "text/plain",
                use_container_width=True,
            )

        if st.session_state.get("research_summary"):
            summary = st.session_state["research_summary"]
            summary_text = (
                "ENGLISH SUMMARY\n\n" + summary.get("english", "") +
                "\n\nROMAN URDU SUMMARY\n\n" + summary.get("roman_urdu", "")
            )
            st.download_button(
                "Download research summary (.txt)",
                summary_text.encode("utf-8"),
                f"{stem}_research_summary.txt",
                "text/plain",
                use_container_width=True,
            )

        st.download_button(
            "Download raw timestamped transcript (.txt)",
            timestamped_text(segments).encode("utf-8"),
            f"{stem}_raw_timestamped.txt",
            "text/plain",
            use_container_width=True,
        )
        st.download_button(
            "Download raw subtitles (.srt)",
            transcript_to_srt(segments).encode("utf-8"),
            f"{stem}_raw.srt",
            "application/x-subrip",
            use_container_width=True,
        )
        st.download_button(
            "Download raw research table (.csv)",
            segments_to_csv(segments).encode("utf-8-sig"),
            f"{stem}_segments.csv",
            "text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download validation report (.json)",
            json.dumps(validation_json, ensure_ascii=False, indent=2).encode("utf-8"),
            f"{stem}_validation.json",
            "application/json",
            use_container_width=True,
        )
