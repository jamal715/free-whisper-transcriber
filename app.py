from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import shutil
import tempfile

import streamlit as st

from cloud_transcriber import ACCURACY_MODEL, FAST_MODEL, transcribe_chunks, translate_chunks
from media import MAX_CHUNK_BYTES, probe_duration, split_audio_lossless
from openai_validator import transcribe_for_validation
from quality import assess_dual, overall_health
from summarizer import extractive_summary
from utils import transcript_to_srt, format_seconds

PRIMARY = "#004d73"

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
  --ink: #121417;
  --muted: #68717c;
  --line: #dce2e6;
  --soft: #f5f7f8;
}}
[data-testid="stSidebar"] {{display:none;}}
[data-testid="stHeader"] {{background:rgba(255,255,255,.94);}}
.stApp {{background:#ffffff; color:var(--ink);}}
.block-container {{max-width:1500px; padding:1rem 2rem 3rem;}}
h1,h2,h3,p,label {{font-family:Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;}}
.q-header {{display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--line); padding:4px 4px 18px; margin-bottom:24px;}}
.q-brand {{display:flex; align-items:center; gap:16px; min-width:0;}}
.q-logo {{width:58px; height:58px; border:5px solid var(--q); border-radius:50%; color:var(--q); font-size:38px; line-height:47px; font-weight:800; text-align:center; box-shadow:inset 0 0 0 5px #dcebf1; font-family:Arial,sans-serif; transform:rotate(-8deg);}}
.q-brand-name {{font-size:34px; font-weight:750; letter-spacing:-1px; color:#101318;}}
.q-tagline {{font-size:16px; color:#46515c; margin-left:6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}}
.q-badge {{border:1px solid #cad4da; border-radius:10px; padding:10px 14px; color:var(--q); background:#fafcfd; font-weight:600; font-size:14px;}}
.hero {{text-align:center; padding:4px 0 18px;}}
.hero h1 {{font-size:42px; line-height:1.05; letter-spacing:-1.5px; margin:.2rem 0 .55rem;}}
.hero p {{color:#6a7380; font-size:18px; margin:0;}}
div[data-testid="stVerticalBlockBorderWrapper"] {{border-color:var(--line)!important; border-radius:14px!important; box-shadow:0 1px 2px rgba(0,0,0,.025);}}
div[data-testid="stFileUploader"] section {{min-height:205px; border:1.5px dashed #a9c6d4; background:#fbfdfe; border-radius:14px;}}
div[data-testid="stFileUploader"] button {{background:var(--q)!important; color:white!important; border:none!important;}}
.stButton > button[kind="primary"] {{background:var(--q)!important; color:white!important; border:1px solid var(--q)!important; min-height:48px; font-weight:700; border-radius:9px;}}
.stTabs [data-baseweb="tab-list"] {{gap:26px; border-bottom:1px solid var(--line);}}
.stTabs [aria-selected="true"] {{color:var(--q)!important;}}
div[data-testid="stMetricValue"] {{color:#111820; font-size:1.45rem;}}
.q-secure {{padding:14px; background:#f7fafb; border:1px solid var(--line); border-radius:12px; color:#5b6570; font-size:13px; line-height:1.5;}}
.q-health {{border-left:4px solid var(--q); background:#f5fafc; padding:14px 16px; border-radius:8px; margin:6px 0 14px;}}
.q-chunk {{border-bottom:1px solid #e6eaed; padding:12px 2px 14px; margin-bottom:4px;}}
.q-time {{color:var(--q); font-size:12px; font-weight:700;}}
.q-text {{font-size:16px; line-height:1.7; margin-top:4px; color:#15191d;}}
.q-provider {{font-size:11px; color:#8a929a; margin-top:5px;}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="q-header">
  <div class="q-brand">
    <div class="q-logo">Q</div>
    <div class="q-brand-name">QuantOra</div>
    <div style="width:1px;height:36px;background:#d9dfe3;margin:0 4px;"></div>
    <div class="q-tagline">Quantifying economic signals for a sustainable future.</div>
  </div>
  <div class="q-badge">🔒 Research workspace</div>
</div>
""",
    unsafe_allow_html=True,
)


def secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "") or "").strip()
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
        f"[{format_seconds(float(seg.get('start', 0.0)))} → {format_seconds(float(seg.get('end', 0.0)))}] {str(seg.get('text', '')).strip()}"
        for seg in segments
        if str(seg.get("text", "")).strip()
    )


def validated_text(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{format_seconds(float(c.get('start', 0.0)))} → {format_seconds(float(c.get('end', 0.0)))}]\n{str(c.get('selected_text', '')).strip()}"
        for c in chunks
        if str(c.get("selected_text", "")).strip()
    )


saved_groq = secret("GROQ_API_KEY")
saved_openai = secret("OPENAI_API_KEY")

st.markdown(
    """
<div class="hero">
  <h1>QuantOra Research Transcriber</h1>
  <p>Accurate transcription and validation for English–Urdu research interviews.</p>
</div>
""",
    unsafe_allow_html=True,
)

control_col, main_col = st.columns([0.92, 3.35], gap="large")

with control_col:
    with st.container(border=True):
        st.markdown("#### Language")
        language_label = st.selectbox(
            "Language",
            ["Urdu + English (mixed)", "English", "Urdu"],
            index=0,
            label_visibility="collapsed",
        )
        language_map = {"Urdu + English (mixed)": None, "English": "en", "Urdu": "ur"}

        st.markdown("#### Transcription mode")
        mode_label = st.selectbox(
            "Mode",
            ["Maximum confidence", "Research accuracy", "Fast draft"],
            index=0,
            label_visibility="collapsed",
            help="Maximum confidence uses Groq Whisper Large V3 plus an independent OpenAI GPT-Transcribe validation pass when an OpenAI key is configured.",
        )
        model = FAST_MODEL if mode_label == "Fast draft" else ACCURACY_MODEL

        st.markdown("#### Translation")
        make_translation = st.toggle("Create English translation", value=False)

        with st.expander("Advanced"):
            context = st.text_area(
                "Names / places / research terms",
                placeholder="PASCO, IFC, Layyah, company names, technical terms…",
                height=100,
                max_chars=900,
            )
            groq_key = saved_groq or st.text_input("Groq API key", type="password", placeholder="gsk_…").strip()
            openai_key = saved_openai or st.text_input(
                "OpenAI API key (optional)",
                type="password",
                placeholder="sk-…",
                help="Required only for dual-engine Maximum confidence validation.",
            ).strip()

        if saved_groq:
            st.caption("● Groq connected")
        if saved_openai:
            st.caption("● OpenAI validator connected")

    st.markdown(
        """
<div class="q-secure">
<strong>Your recording is not stored by this app.</strong><br>
Temporary processing files are deleted after the run. API keys should be kept in Streamlit Secrets.
</div>
""",
        unsafe_allow_html=True,
    )

with main_col:
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Drag & drop your audio or video file here",
            type=["m4a", "mp3", "mp4", "wav", "flac", "ogg", "webm", "mpeg", "mpga"],
            accept_multiple_files=False,
            help="Designed for interviews up to 120 minutes. Large files are automatically split.",
        )

    ready = uploaded is not None and bool(groq_key)
    if uploaded is not None:
        size_mb = uploaded.size / (1024 * 1024)
        c1, c2, c3 = st.columns(3)
        c1.metric("File", uploaded.name)
        c2.metric("Size", f"{size_mb:.1f} MB")
        if mode_label == "Maximum confidence" and openai_key:
            c3.metric("Verification", "Dual engine")
        elif mode_label == "Maximum confidence":
            c3.metric("Verification", "QA scan only")
        else:
            c3.metric("Model", "Large V3" if model == ACCURACY_MODEL else "Large V3 Turbo")
        if uploaded.size > MAX_CHUNK_BYTES:
            st.caption("Large recording detected — the app will split it losslessly into safe overlapping chunks, transcribe each part, then stitch the interview back together.")

    if not groq_key:
        st.info("Add your Groq key under Advanced to enable transcription.")

    if mode_label == "Maximum confidence" and not openai_key:
        st.warning("Maximum confidence is currently running in single-engine mode. Add an OpenAI API key under Advanced to enable independent chunk-by-chunk validation.")

    start = st.button(
        "Transcribe interview" if ready else "Waiting for file and API key…",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    )

    if start:
        st.session_state.pop("result", None)
        work_dir = Path(tempfile.mkdtemp(prefix="quantora_research_"))
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
                status.write(f"Recording length: {format_seconds(duration)}")
            if duration and duration > 2 * 60 * 60 + 30:
                raise RuntimeError("This workspace is configured for interviews up to 120 minutes.")

            if source_path.stat().st_size <= MAX_CHUNK_BYTES:
                chunks = [{"path": str(source_path), "duration": duration, "offset": 0.0, "keep_after": 0.0, "nominal_end": duration}]
            else:
                progress.progress(4, text="Preparing safe chunks…")
                status.write("Splitting large recording without re-encoding.")
                chunks = split_audio_lossless(str(source_path), str(chunk_dir))

            status.write(f"Prepared {len(chunks)} part{'s' if len(chunks) != 1 else ''}.")

            def transcription_progress(done: int, total: int, message: str):
                ceiling = 52 if (mode_label == "Maximum confidence" and openai_key) else 83
                pct = 6 + int((done / max(1, total)) * ceiling)
                progress.progress(min(88, pct), text=message)

            result = transcribe_chunks(
                chunks=chunks,
                api_key=groq_key,
                language=language_map[language_label],
                model=model,
                context_prompt=context.strip(),
                progress_callback=transcription_progress,
            )

            validated_chunks = []
            verifier_model = None
            chunk_results = result.get("chunk_results", [])

            for idx, chunk_result in enumerate(chunk_results):
                verifier_text = None
                verifier_error = None
                if mode_label == "Maximum confidence" and openai_key:
                    progress.progress(60 + int((idx / max(1, len(chunk_results))) * 30), text=f"Independent validation: part {idx + 1} of {len(chunk_results)}…")
                    try:
                        verification = transcribe_for_validation(
                            path=chunks[idx]["path"],
                            api_key=openai_key,
                            context_prompt=("This is a Pakistani research interview. Speech may naturally code-switch between Urdu and English. Preserve what is spoken; do not translate or invent. " + context.strip())[:900],
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
                    assessment["reasons"] = list(assessment.get("reasons", [])) + ["independent validator unavailable for this chunk"]
                    if assessment["status"] == "passed":
                        assessment["status"] = "review"
                        assessment["score"] = min(assessment["score"], 78)
                validated_chunks.append(assessment)

            health = overall_health(validated_chunks)
            result["validated_chunks"] = validated_chunks
            result["health"] = health
            result["verifier_model"] = verifier_model
            result["validated_text"] = "\n\n".join(c.get("selected_text", "").strip() for c in validated_chunks if c.get("selected_text", "").strip())

            if make_translation:
                progress.progress(92, text="Creating separate English translation…")
                result["translation"] = translate_chunks(chunks=chunks, api_key=groq_key, context_prompt=context.strip())

            result["source_name"] = uploaded.name
            result["requested_model"] = model
            st.session_state["result"] = result
            progress.progress(100, text="Interview ready")
            if health["failed"] or health["review"]:
                status.update(label="Completed — review flagged sections", state="complete", expanded=False)
            else:
                status.update(label="Completed — high confidence", state="complete", expanded=False)

        except Exception as exc:
            status.update(label="Transcription failed", state="error", expanded=True)
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
<strong>Transcript health: {status_text}</strong> &nbsp;·&nbsp;
Score {health.get('score', 0)}/100 &nbsp;·&nbsp;
{health.get('passed', 0)} passed &nbsp;·&nbsp;
{health.get('review', 0)} review &nbsp;·&nbsp;
{health.get('failed', 0)} failed
</div>
""",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration", format_seconds(duration))
    m2.metric("Parts", int(result.get("parts", 1)))
    m3.metric("Reliability", f"{health.get('score', 0)}/100")
    m4.metric("Validation", "Dual engine" if result.get("verifier_model") else "Single engine + QA")

    tab_labels = ["Transcript"]
    if result.get("translation"):
        tab_labels.append("Translation")
    tab_labels += ["Validation", "Summary", "Downloads"]
    tabs = st.tabs(tab_labels)
    tab_i = 0

    with tabs[tab_i]:
        tab_i += 1
        for chunk in validated_chunks:
            state = chunk.get("status", "review")
            state_icon = "✓" if state == "passed" else "!" if state == "review" else "×"
            provider = chunk.get("selected_provider", "Groq")
            safe_text = str(chunk.get("selected_text", "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(
                f"""
<div class="q-chunk">
  <div class="q-time">{format_seconds(chunk.get('start', 0.0))} — {format_seconds(chunk.get('end', 0.0))} &nbsp; {state_icon} {state.upper()} · {chunk.get('score', 0)}/100</div>
  <div class="q-text">{safe_text}</div>
  <div class="q-provider">{provider}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        with st.expander("Raw detailed timestamps"):
            st.text_area("Raw segment transcript", value=timestamped_text(segments), height=500, label_visibility="collapsed")

    if result.get("translation"):
        with tabs[tab_i]:
            tab_i += 1
            translated = result["translation"]
            st.text_area("English translation", value=timestamped_text(translated.get("segments", [])) or translated.get("text", ""), height=600, label_visibility="collapsed")

    with tabs[tab_i]:
        tab_i += 1
        rows = []
        for chunk in validated_chunks:
            similarity = chunk.get("similarity")
            rows.append({
                "time": f"{format_seconds(chunk.get('start', 0))}–{format_seconds(chunk.get('end', 0))}",
                "status": chunk.get("status"),
                "score": chunk.get("score"),
                "engine agreement": "" if similarity is None else f"{similarity:.0%}",
                "selected": chunk.get("selected_provider"),
                "reason": ", ".join(chunk.get("reasons", [])),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("A fluent transcript is not automatically trustworthy. This panel exposes disagreement between engines and corruption checks so you can listen back only where needed.")

    with tabs[tab_i]:
        tab_i += 1
        summary_source = result.get("translation", {}).get("text", "") if result.get("translation") else result.get("validated_text", "")
        summary_length = st.slider("Summary length", 4, 16, 8)
        summary = extractive_summary(summary_source, max_sentences=summary_length)
        if summary:
            st.markdown(summary)
            st.caption("If Translation is enabled, this summary is based on the English translation. Otherwise it preserves the source language.")
        else:
            st.info("Not enough validated text to summarize.")

    with tabs[tab_i]:
        stem = Path(source_name).stem
        validation_json = {
            "source_file": source_name,
            "health": health,
            "model": result.get("model"),
            "verifier_model": result.get("verifier_model"),
            "validated_chunks": validated_chunks,
        }
        st.download_button("Download validated transcript (.txt)", validated_text(validated_chunks).encode("utf-8"), f"{stem}_validated_transcript.txt", "text/plain", use_container_width=True)
        st.download_button("Download raw timestamped transcript (.txt)", timestamped_text(segments).encode("utf-8"), f"{stem}_raw_timestamped.txt", "text/plain", use_container_width=True)
        st.download_button("Download raw subtitles (.srt)", transcript_to_srt(segments).encode("utf-8"), f"{stem}_raw.srt", "application/x-subrip", use_container_width=True)
        st.download_button("Download raw research table (.csv)", segments_to_csv(segments).encode("utf-8-sig"), f"{stem}_segments.csv", "text/csv", use_container_width=True)
        st.download_button("Download validation report (.json)", json.dumps(validation_json, ensure_ascii=False, indent=2).encode("utf-8"), f"{stem}_validation.json", "application/json", use_container_width=True)
