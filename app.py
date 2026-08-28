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
from quality import assess_single, overall_health
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
  --q:{PRIMARY};
  --ink:#111827;
  --muted:#667085;
  --faint:#89939e;
  --line:#d9e1e6;
  --soft:#f7fafb;
  --green:#18794e;
  --amber:#9a6700;
  --red:#b42318;
}}
[data-testid="stSidebar"]{{display:none;}}
[data-testid="stHeader"]{{background:rgba(255,255,255,.98);height:3.25rem;}}
.stApp{{background:#ffffff;color:var(--ink);}}
.block-container{{max-width:1480px;padding:4.6rem 2.2rem 3rem!important;}}
html,body,[class*="css"]{{font-family:Inter,"Avenir Next","Segoe UI",system-ui,sans-serif;}}
h1,h2,h3,h4,p,label,span,div{{color:inherit;}}

.q-top{{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding:0 0 17px;margin-bottom:24px;}}
.q-brand{{display:flex;align-items:center;gap:14px;min-width:0;}}
.q-brand img{{width:54px!important;height:54px!important;object-fit:contain!important;border-radius:10px!important;}}
.q-name{{font-size:29px;font-weight:800;letter-spacing:-.8px;color:var(--ink);}}
.q-divider{{width:1px;height:34px;background:var(--line);margin:0 4px;}}
.q-tag{{font-size:13px;color:#52606b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:520px;}}
.q-pill{{border:1px solid #cbd8de;border-radius:10px;padding:9px 13px;color:var(--q);background:#fbfcfd;font-size:12px;font-weight:750;}}

.q-hero{{text-align:center;padding:10px 0 23px;}}
.q-hero h1{{font-size:40px;line-height:1.08;letter-spacing:-1.35px;margin:0 0 9px;color:var(--ink);font-weight:800;}}
.q-hero p{{font-size:15px;color:var(--muted);margin:0 auto;max-width:820px;line-height:1.55;}}
.q-badge-row{{display:flex;justify-content:center;gap:8px;margin-top:13px;flex-wrap:wrap;}}
.q-badge{{font-size:11px;color:var(--q);font-weight:750;background:#f1f7fa;border:1px solid #d6e5ec;padding:7px 10px;border-radius:999px;}}

.q-panel-title{{font-size:13px;font-weight:800;text-transform:uppercase;letter-spacing:.07em;color:#344054;margin:1px 0 8px;}}
.q-note{{background:#f8fafb;border:1px solid var(--line);border-radius:12px;padding:13px 14px;color:#5d6973;font-size:12px;line-height:1.55;margin-top:14px;}}
.q-ready-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin:12px 0 14px;}}
.q-ready{{background:#fafcfd;border:1px solid var(--line);border-radius:11px;padding:11px 12px;}}
.q-ready-label{{font-size:10px;color:#7a8691;text-transform:uppercase;letter-spacing:.07em;font-weight:750;}}
.q-ready-value{{font-size:14px;color:var(--ink);font-weight:750;margin-top:3px;}}
.q-ok{{color:var(--green)!important;}}
.q-warn{{color:var(--amber)!important;}}

.q-pipeline{{background:#f7fbfc;border:1px solid #d6e5eb;border-radius:12px;padding:13px 15px;margin:12px 0;}}
.q-pipeline strong{{color:var(--q);}}
.q-pipeline small{{color:var(--muted);}}
.q-health{{display:flex;align-items:center;justify-content:space-between;gap:18px;background:#f5fafc;border:1px solid #d6e5eb;border-left:4px solid var(--q);border-radius:11px;padding:14px 16px;margin:14px 0;}}
.q-health-title{{font-size:16px;color:var(--ink);font-weight:800;}}
.q-health-sub{{font-size:12px;color:var(--muted);margin-top:3px;}}
.q-score{{font-size:28px;font-weight:850;color:var(--q);letter-spacing:-1px;}}
.q-chunk{{border-bottom:1px solid #e7ecef;padding:13px 2px 15px;}}
.q-time{{font-size:11px;color:var(--q);font-weight:800;}}
.q-text{{font-size:15px;line-height:1.72;color:#171a1d;margin-top:5px;}}
.q-provider{{font-size:10px;color:#8b949c;margin-top:6px;}}
.q-silence{{font-size:13px;color:#7a8691;font-style:italic;margin-top:5px;}}

/* Native Streamlit controls */
div[data-testid="stVerticalBlockBorderWrapper"]{{background:#fff!important;border:1px solid var(--line)!important;border-radius:14px!important;box-shadow:0 1px 2px rgba(16,24,40,.03)!important;}}
div[data-testid="stFileUploader"] section{{min-height:215px;border:1.4px dashed #a9c6d4!important;background:#fbfdfe!important;border-radius:13px!important;}}
div[data-testid="stFileUploader"] button{{background:var(--q)!important;color:#fff!important;border:none!important;border-radius:9px!important;font-weight:750!important;}}
[data-testid="stFileUploaderDropzoneInstructions"] span,[data-testid="stFileUploaderDropzoneInstructions"] small{{color:#4f5d68!important;}}
.stButton>button[kind="primary"]{{background:var(--q)!important;color:#fff!important;border:1px solid var(--q)!important;min-height:50px;font-weight:800;border-radius:9px;}}
.stButton>button[kind="secondary"]{{background:#fff!important;color:var(--q)!important;border:1px solid #b9ccd5!important;font-weight:750!important;}}
.stButton>button:disabled{{opacity:.50!important;}}

[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-baseweb="textarea"],textarea,input{{
  background:#fff!important;color:#111827!important;-webkit-text-fill-color:#111827!important;border-color:#cfd8de!important;
}}
[data-baseweb="select"] span,[data-baseweb="select"] svg,[data-baseweb="input"] input,[data-baseweb="textarea"] textarea{{
  color:#111827!important;-webkit-text-fill-color:#111827!important;
}}
input::placeholder,textarea::placeholder{{color:#7b8790!important;-webkit-text-fill-color:#7b8790!important;opacity:1!important;}}
[data-baseweb="popover"],[role="listbox"],[role="option"]{{background:#fff!important;color:#111827!important;}}
[role="option"] *,[role="listbox"] *{{color:#111827!important;-webkit-text-fill-color:#111827!important;}}

.stTabs [data-baseweb="tab-list"]{{gap:25px;border-bottom:1px solid var(--line);}}
.stTabs [data-baseweb="tab"]{{color:#475467!important;font-weight:700!important;}}
.stTabs [aria-selected="true"]{{color:var(--q)!important;border-bottom-color:var(--q)!important;}}
div[data-testid="stMetric"]{{background:#fafcfd;border:1px solid var(--line);border-radius:11px;padding:11px 13px;}}
div[data-testid="stMetricLabel"]{{color:#667085!important;}}
div[data-testid="stMetricValue"]{{color:#111827!important;font-size:1.38rem;}}
div[data-testid="stAlert"]{{border-radius:10px;}}
div[data-testid="stDataFrame"]{{border:1px solid var(--line);border-radius:11px;overflow:hidden;}}
hr{{border-color:var(--line)!important;}}

@media(max-width:1000px){{
 .q-tag,.q-divider{{display:none}}
 .q-ready-grid{{grid-template-columns:1fr 1fr}}
 .block-container{{padding:4.2rem 1rem 2rem!important}}
 .q-hero h1{{font-size:31px}}
}}
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
  <div class="q-pill">Research workspace</div>
</div>
<div class="q-hero">
  <h1>QuantOra Research Transcriber</h1>
  <p>Accuracy-first transcription for English–Urdu research interviews. Each listening window is processed independently, and low-confidence speech is withheld rather than guessed.</p>
  <div class="q-badge-row">
    <div class="q-badge">Whisper Large V3</div>
    <div class="q-badge">Anti-hallucination QA</div>
    <div class="q-badge">Up to 120 minutes</div>
    <div class="q-badge">Instant local summary</div>
  </div>
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
        f"[{format_seconds(float(s.get('start', 0)))} → {format_seconds(float(s.get('end', 0)))}] {str(s.get('text', '')).strip()}"
        for s in segments if str(s.get("text", "")).strip()
    )


def validated_text(chunks: list[dict]) -> str:
    rows = []
    for c in chunks:
        state = c.get("status")
        start = format_seconds(float(c.get("start", 0)))
        end = format_seconds(float(c.get("end", 0)))
        if state == "silence":
            rows.append(f"[{start} → {end}] [silence / no confident speech]")
        elif c.get("selected_text"):
            rows.append(f"[{start} → {end}]\n{str(c.get('selected_text')).strip()}")
        elif state == "review":
            rows.append(f"[{start} → {end}] [LOW-CONFIDENCE AUDIO — wording withheld to avoid hallucination]")
        else:
            rows.append(f"[{start} → {end}] [REVIEW REQUIRED — transcription unavailable]")
    return "\n\n".join(rows)


def translation_text(result: dict) -> str:
    return "\n\n".join(
        f"[{format_seconds(float(c.get('start', 0)))} → {format_seconds(float(c.get('end', 0)))}]\n{str(c.get('text', '')).strip()}"
        for c in result.get("chunks", []) if str(c.get("text", "")).strip()
    )


def segments_to_csv(segments: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["start_seconds", "end_seconds", "start", "end", "text", "review_flag"])
    for s in segments:
        w.writerow([
            f"{float(s.get('start', 0)):.3f}",
            f"{float(s.get('end', 0)):.3f}",
            format_seconds(float(s.get("start", 0))),
            format_seconds(float(s.get("end", 0))),
            str(s.get("text", "")).strip(),
            "YES" if s.get("review_flag") else "",
        ])
    return buf.getvalue()


def status_card(label: str, value: str, ok: bool = False, warn: bool = False) -> str:
    cls = "q-ok" if ok else "q-warn" if warn else ""
    return f'<div class="q-ready"><div class="q-ready-label">{html.escape(label)}</div><div class="q-ready-value {cls}">{html.escape(value)}</div></div>'


saved_groq = secret("GROQ_API_KEY")
if "session_groq_key" not in st.session_state:
    st.session_state.session_groq_key = ""

left, main = st.columns([0.24, 0.76], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="q-panel-title">Interview language</div>', unsafe_allow_html=True)
        language_label = st.selectbox(
            "Language",
            ["Urdu + English (mixed)", "Mostly Urdu", "Mostly English"],
            index=0,
            label_visibility="collapsed",
            help="Use Mixed for normal Pakistani code-switching.",
        )
        language_map = {"Urdu + English (mixed)": None, "Mostly Urdu": "ur", "Mostly English": "en"}

        st.markdown('<div class="q-panel-title" style="margin-top:20px">Engine mode</div>', unsafe_allow_html=True)
        mode_label = st.selectbox(
            "Mode",
            ["Research accuracy", "Fast draft"],
            index=0,
            label_visibility="collapsed",
        )
        model = ACCURACY_MODEL if mode_label == "Research accuracy" else FAST_MODEL

        st.markdown('<div class="q-panel-title" style="margin-top:20px">Context</div>', unsafe_allow_html=True)
        context = st.text_area(
            "Exact names / places / acronyms only",
            placeholder="HBL, PASCO, Layyah, company names, acronyms…",
            height=88,
            max_chars=500,
            help="Optional. Keep this to exact terms only. It is not used as a narrative prompt.",
        )

        st.markdown('<div class="q-panel-title" style="margin-top:20px">API</div>', unsafe_allow_html=True)
        if saved_groq:
            st.success("Groq connected from Streamlit Secrets")
        else:
            st.text_input("Groq API key", type="password", placeholder="gsk_…", key="groq_key_input")
            if st.button("Apply Groq key", use_container_width=True):
                st.session_state.session_groq_key = str(st.session_state.get("groq_key_input", "") or "").strip()
            if st.session_state.session_groq_key:
                st.success("Groq key active for this session")

    st.markdown(
        '<div class="q-note"><strong style="color:#111827">Research rule</strong><br>'
        'The validated transcript favors precision over recall. Silence is not a failure. '
        'Low-confidence wording can be withheld rather than guessed. Raw ASR remains available separately for audit.</div>',
        unsafe_allow_html=True,
    )


groq_key = saved_groq or st.session_state.session_groq_key

with main:
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Upload interview audio or video",
            type=["m4a", "mp3", "mp4", "wav", "flac", "ogg", "webm", "mpeg", "mpga"],
            accept_multiple_files=False,
            help="Up to 120 minutes. MP3/M4A recommended.",
        )

    file_ready = uploaded is not None
    key_ready = bool(groq_key)
    ready = file_ready and key_ready

    readiness = (
        '<div class="q-ready-grid">'
        + status_card("Recording", "Ready" if file_ready else "Waiting", ok=file_ready)
        + status_card("Speech engine", "Connected" if key_ready else "API key needed", ok=key_ready, warn=not key_ready)
        + status_card("Accuracy", "Large V3" if model == ACCURACY_MODEL else "Large V3 Turbo", ok=True)
        + status_card("Post-processing", "On demand", ok=True)
        + "</div>"
    )
    st.markdown(readiness, unsafe_allow_html=True)

    if uploaded is not None:
        size_mb = uploaded.size / (1024 * 1024)
        a, b, c = st.columns(3)
        a.metric("Recording", uploaded.name)
        b.metric("Upload", f"{size_mb:.1f} MB")
        c.metric("Pipeline", "Independent 60-sec windows")

    start = st.button(
        "Run research transcription" if ready else "Waiting for recording + Groq key",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    )

    pipeline_box = st.empty()

    if not st.session_state.get("result") and not start:
        tabs = st.tabs(["Transcript", "Translation", "Quality", "Summary", "Downloads"])
        with tabs[0]:
            st.caption("The validated transcript appears here when the speech pass is complete.")
        with tabs[1]:
            st.caption("Translation is optional and runs only after transcription, so it cannot delay your transcript.")
        with tabs[2]:
            st.caption("Silence, review, failures and confidence signals are separated here.")
        with tabs[3]:
            st.caption("Summary is instant and extractive: it copies representative text from passed windows only.")
        with tabs[4]:
            st.caption("Download validated text, raw timestamps, subtitles, CSV and QA report here.")

    if start:
        st.session_state.pop("result", None)
        work_dir = Path(tempfile.mkdtemp(prefix="quantora_research_"))
        suffix = Path(uploaded.name).suffix.lower() or ".m4a"
        source_path = work_dir / f"source{suffix}"
        chunk_dir = work_dir / "chunks"
        progress = st.progress(0, text="Preparing recording…")

        def pipeline(title: str, detail: str) -> None:
            pipeline_box.markdown(
                f'<div class="q-pipeline"><strong>{html.escape(title)}</strong><br><small>{html.escape(detail)}</small></div>',
                unsafe_allow_html=True,
            )

        try:
            pipeline("Stage 1 · Audio intake", "Receiving the recording and checking duration.")
            with source_path.open("wb") as out:
                uploaded.seek(0)
                shutil.copyfileobj(uploaded, out, length=1024 * 1024)

            duration = probe_duration(str(source_path))
            if not duration:
                raise RuntimeError("Could not read recording duration. MP3 or M4A is recommended.")
            if duration > MAX_DURATION_SECONDS + 0.5:
                raise RuntimeError(f"Maximum supported duration is 120 minutes. This recording is {format_seconds(duration)}.")

            pipeline("Stage 2 · Accuracy preparation", "Creating short lossless listening windows.")
            if duration <= 300 and source_path.stat().st_size <= MAX_CHUNK_BYTES:
                chunks = [{
                    "path": str(source_path),
                    "duration": duration,
                    "offset": 0.0,
                    "keep_after": 0.0,
                    "nominal_end": duration,
                    "overlap_seconds": 0.0,
                }]
            else:
                chunks = split_audio_lossless(str(source_path), str(chunk_dir))

            prompt = context.strip()[:500]

            def progress_cb(done: int, total: int, message: str) -> None:
                progress.progress(min(86, 5 + int((done / max(1, total)) * 81)), text=message)
                pipeline("Stage 3 · Speech recognition", f"{message} · {done}/{total} windows complete")

            result = transcribe_chunks(
                chunks=chunks,
                api_key=groq_key,
                language=language_map[language_label],
                model=model,
                context_prompt=prompt,
                progress_callback=progress_cb,
                chunk_callback=None,
            )

            pipeline("Stage 4 · Integrity scan", "Withholding corrupted or low-confidence wording before downstream use.")
            validated: list[dict] = []
            for cr in result.get("chunk_results", []):
                assessment = assess_single(
                    cr.get("text", ""),
                    api_error=cr.get("error"),
                    avg_no_speech_prob=cr.get("avg_no_speech_prob"),
                    max_no_speech_prob=cr.get("max_no_speech_prob"),
                    avg_logprob=cr.get("avg_logprob"),
                    segment_count=int(cr.get("raw_segment_count", cr.get("segment_count", 0)) or 0),
                )
                assessment.update({
                    "index": cr.get("index"),
                    "start": float(cr.get("keep_after", cr.get("offset", 0.0))),
                    "end": float(cr.get("end", 0.0)),
                    "groq_text": cr.get("text", ""),
                    "avg_no_speech_prob": cr.get("avg_no_speech_prob"),
                    "avg_logprob": cr.get("avg_logprob"),
                })
                validated.append(assessment)

            health = overall_health(validated)
            result["validated_chunks"] = validated
            result["health"] = health
            result["source_name"] = uploaded.name

            st.session_state["result"] = result
            progress.progress(100, text="Research transcript ready")
            if result.get("stopped_reason"):
                pipeline("Partial transcript preserved", result["stopped_reason"])
            else:
                pipeline("Complete ✓", "Transcript and quality map are ready. Translation is optional and separate.")

        except Exception as exc:
            pipeline("Processing stopped", str(exc))
            st.error(str(exc))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


result = st.session_state.get("result")
if result:
    validated = result.get("validated_chunks", [])
    health = result.get("health", {})
    segments = result.get("segments", [])
    duration = float(result.get("duration", 0.0) or 0.0)
    source_name = result.get("source_name", "interview")

    st.markdown("---")
    h_status = str(health.get("status", "unknown")).title()
    st.markdown(
        f'<div class="q-health"><div><div class="q-health-title">Transcript health · {html.escape(h_status)}</div>'
        f'<div class="q-health-sub">{health.get("passed",0)} passed · {health.get("review",0)} review · '
        f'{health.get("failed",0)} failed · {health.get("silence",0)} silence</div></div>'
        f'<div class="q-score">{health.get("score",0)}/100</div></div>',
        unsafe_allow_html=True,
    )

    if result.get("stopped_reason"):
        st.warning(result["stopped_reason"])

    processed_parts = int(result.get("processed_parts", result.get("parts", 1)))
    total_parts = int(result.get("parts", 1))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration", format_seconds(duration))
    m2.metric("Processed", f"{processed_parts}/{total_parts}")
    m3.metric("Speech reliability", f"{health.get('score',0)}/100")
    m4.metric("Engine", "Whisper Large V3" if result.get("model") == ACCURACY_MODEL else "Large V3 Turbo")

    tabs = st.tabs(["Transcript", "Translation", "Quality", "Summary", "Downloads"])

    with tabs[0]:
        for c in validated:
            state = c.get("status", "review")
            start_t = format_seconds(float(c.get("start", 0)))
            end_t = format_seconds(float(c.get("end", 0)))

            if state == "silence":
                st.markdown(
                    f'<div class="q-chunk"><div class="q-time">{start_t} — {end_t} · SILENCE</div>'
                    '<div class="q-silence">No confident speech detected.</div></div>',
                    unsafe_allow_html=True,
                )
                continue

            icon = "✓" if state == "passed" else "!" if state == "review" else "×"
            safe = html.escape(str(c.get("selected_text", "") or "")).replace("\n", "<br>")
            if not safe and state == "review":
                safe = "[Low-confidence audio — wording withheld to avoid hallucination.]"
            elif not safe:
                safe = "[Transcription unavailable — listen back to this window.]"

            score = "—" if c.get("score") is None else c.get("score")
            st.markdown(
                f'<div class="q-chunk"><div class="q-time">{start_t} — {end_t} · {icon} {state.upper()} · {score}/100</div>'
                f'<div class="q-text">{safe}</div><div class="q-provider">Groq Whisper Large V3</div></div>',
                unsafe_allow_html=True,
            )

        with st.expander("Raw ASR transcript · unverified"):
            st.caption("This preserves the speech engine's raw wording for audit. Use the validated transcript above for research.")
            st.text_area("Raw", value=timestamped_text(segments), height=500, label_visibility="collapsed")

    with tabs[1]:
        if result.get("translation"):
            st.text_area("English translation", value=translation_text(result["translation"]), height=620, label_visibility="collapsed")
            st.caption("Only PASSED source windows are translated. REVIEW/FAILED wording is withheld.")
        else:
            st.caption("Translation is deliberately separate so your transcript finishes first.")
            if st.button("Generate English translation from passed windows", use_container_width=True, key="translate_btn"):
                with st.spinner("Translating validated speech…"):
                    result["translation"] = translate_validated_chunks(
                        validated_chunks=validated,
                        api_key=groq_key,
                        progress_callback=None,
                    )
                    st.session_state["result"] = result
                    st.rerun()

    with tabs[2]:
        rows = []
        for c in validated:
            rows.append({
                "time": f"{format_seconds(float(c.get('start',0)))}–{format_seconds(float(c.get('end',0)))}",
                "state": c.get("status"),
                "score": "—" if c.get("score") is None else c.get("score"),
                "no-speech": "" if c.get("avg_no_speech_prob") is None else f"{float(c.get('avg_no_speech_prob')):.2f}",
                "log-prob": "" if c.get("avg_logprob") is None else f"{float(c.get('avg_logprob')):.2f}",
                "reason": ", ".join(c.get("reasons", [])),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("SILENCE is neutral. REVIEW/FAILED windows are the only places that need listening back.")

    with tabs[3]:
        summary = build_research_summary(validated_chunks=validated, api_key="")
        st.markdown("### Source-grounded highlights")
        st.markdown(summary.get("english", "") or "_No passed transcript windows were available._")
        st.caption("Instant extractive summary: representative wording is copied from PASSED windows only. No text-generation API is used.")

    with tabs[4]:
        stem = Path(source_name).stem
        report = {
            "source_file": source_name,
            "health": health,
            "model": result.get("model"),
            "stopped_reason": result.get("stopped_reason"),
            "validated_chunks": validated,
        }
        st.download_button(
            "Validated transcript · TXT",
            validated_text(validated).encode("utf-8"),
            f"{stem}_validated_transcript.txt",
            "text/plain",
            use_container_width=True,
        )
        if result.get("translation"):
            st.download_button(
                "English translation · TXT",
                translation_text(result["translation"]).encode("utf-8"),
                f"{stem}_english_translation.txt",
                "text/plain",
                use_container_width=True,
            )
        st.download_button(
            "Raw timestamped transcript · TXT",
            timestamped_text(segments).encode("utf-8"),
            f"{stem}_raw_timestamped.txt",
            "text/plain",
            use_container_width=True,
        )
        st.download_button(
            "Subtitles · SRT",
            transcript_to_srt(segments).encode("utf-8"),
            f"{stem}_raw.srt",
            "application/x-subrip",
            use_container_width=True,
        )
        st.download_button(
            "Research table · CSV",
            segments_to_csv(segments).encode("utf-8-sig"),
            f"{stem}_segments.csv",
            "text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Quality report · JSON",
            json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
            f"{stem}_quality.json",
            "application/json",
            use_container_width=True,
        )
