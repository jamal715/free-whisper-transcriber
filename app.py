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
  --q-deep:#003b59;
  --q-dark:#002f47;
  --panel:#074664;
  --panel2:#0a415b;
  --line:rgba(255,255,255,.14);
  --text:#f6f8fa;
  --muted:#b6c6cf;
  --faint:#8fa8b5;
  --accent:#78d1f3;
  --green:#6fe0aa;
  --amber:#ffd37a;
  --red:#ff8f91;
}}
[data-testid="stSidebar"] {{display:none;}}
[data-testid="stHeader"] {{background:rgba(0,47,71,.98); height:3.25rem;}}
[data-testid="stToolbar"] {{color:white;}}
.stApp {{background:var(--q); color:var(--text);}}
.block-container {{max-width:1540px; padding:4.6rem 2.4rem 4rem!important;}}
html,body,[class*="css"] {{font-family:Inter,"Avenir Next","Segoe UI",system-ui,sans-serif;}}
h1,h2,h3,h4,p,label,span,div {{color:inherit;}}

.q-top {{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding:0 0 18px;margin-bottom:24px;}}
.q-brand {{display:flex;align-items:center;gap:14px;min-width:0;}}
.q-brand img {{width:56px!important;height:56px!important;object-fit:contain!important;border-radius:12px!important;box-shadow:0 8px 24px rgba(0,0,0,.18);}}
.q-name {{font-size:29px;font-weight:800;letter-spacing:-.8px;color:white;}}
.q-divider {{width:1px;height:35px;background:var(--line);margin:0 4px;}}
.q-tag {{font-size:13px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:530px;}}
.q-pill {{border:1px solid var(--line);border-radius:999px;padding:9px 14px;color:#d9f3ff;background:rgba(255,255,255,.06);font-size:12px;font-weight:700;}}

.q-hero {{padding:10px 0 24px;display:flex;align-items:flex-end;justify-content:space-between;gap:20px;}}
.q-hero h1 {{font-size:42px;line-height:1.05;letter-spacing:-1.6px;margin:0 0 8px;color:white;}}
.q-hero p {{font-size:16px;color:var(--muted);margin:0;max-width:760px;}}
.q-engine {{font-size:12px;color:var(--accent);font-weight:700;background:rgba(120,209,243,.08);border:1px solid rgba(120,209,243,.25);padding:9px 12px;border-radius:9px;white-space:nowrap;}}

.q-panel-title {{font-size:13px;font-weight:750;text-transform:uppercase;letter-spacing:.08em;color:#d9edf7;margin:1px 0 8px;}}
.q-note {{background:rgba(255,255,255,.055);border:1px solid var(--line);border-radius:12px;padding:13px 14px;color:var(--muted);font-size:12px;line-height:1.55;}}
.q-ready-grid {{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:12px 0 14px;}}
.q-ready {{background:rgba(255,255,255,.055);border:1px solid var(--line);border-radius:12px;padding:12px 13px;}}
.q-ready-label {{font-size:10px;color:var(--faint);text-transform:uppercase;letter-spacing:.08em;font-weight:700;}}
.q-ready-value {{font-size:14px;color:white;font-weight:750;margin-top:3px;}}
.q-ok {{color:var(--green)!important;}}
.q-warn {{color:var(--amber)!important;}}

.q-pipeline {{background:rgba(0,0,0,.14);border:1px solid var(--line);border-radius:13px;padding:14px 16px;margin:12px 0;}}
.q-pipeline strong {{color:white;}}
.q-pipeline small {{color:var(--muted);}}
.q-live {{background:var(--q-deep);border:1px solid var(--line);border-radius:14px;padding:16px;margin-top:12px;max-height:520px;overflow-y:auto;}}
.q-live-title {{font-weight:800;color:white;margin-bottom:4px;}}
.q-live-sub {{font-size:12px;color:var(--muted);margin-bottom:10px;}}
.q-live-part {{padding:10px 0;border-top:1px solid rgba(255,255,255,.09);}}
.q-live-time {{font-size:10px;color:var(--accent);font-weight:750;letter-spacing:.03em;}}
.q-live-text {{font-size:14px;color:#eef5f8;line-height:1.65;margin-top:4px;}}

.q-health {{display:flex;align-items:center;justify-content:space-between;gap:18px;background:var(--q-dark);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin:16px 0;}}
.q-health-title {{font-size:16px;color:white;font-weight:800;}}
.q-health-sub {{font-size:12px;color:var(--muted);margin-top:3px;}}
.q-score {{font-size:29px;font-weight:850;color:white;letter-spacing:-1px;}}
.q-chunk {{border-bottom:1px solid rgba(255,255,255,.10);padding:14px 3px 16px;}}
.q-time {{font-size:11px;color:var(--accent);font-weight:800;}}
.q-text {{font-size:15px;line-height:1.75;color:#f1f6f8;margin-top:5px;}}
.q-provider {{font-size:10px;color:var(--faint);margin-top:6px;}}
.q-silence {{font-size:13px;color:var(--faint);font-style:italic;margin-top:5px;}}

/* Native Streamlit controls */
div[data-testid="stVerticalBlockBorderWrapper"] {{background:rgba(0,0,0,.08)!important;border:1px solid var(--line)!important;border-radius:15px!important;box-shadow:none!important;}}
div[data-testid="stFileUploader"] section {{min-height:210px;border:1.4px dashed rgba(169,219,239,.55)!important;background:rgba(0,0,0,.08)!important;border-radius:13px!important;}}
div[data-testid="stFileUploader"] button {{background:#e9f6fb!important;color:#003b59!important;border:none!important;border-radius:9px!important;font-weight:750!important;}}
[data-testid="stFileUploaderDropzoneInstructions"] span,[data-testid="stFileUploaderDropzoneInstructions"] small {{color:#d9e8ee!important;}}
.stButton>button[kind="primary"] {{background:#e8f6fb!important;color:#003b59!important;border:none!important;min-height:50px;font-weight:800;border-radius:10px;}}
.stButton>button[kind="secondary"] {{background:rgba(255,255,255,.08)!important;color:white!important;border:1px solid var(--line)!important;}}
.stButton>button:disabled {{opacity:.45!important;}}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,textarea {{background:rgba(0,0,0,.13)!important;color:white!important;border-color:var(--line)!important;}}
[data-baseweb="select"] span,input,textarea {{color:white!important;}}
.stTabs [data-baseweb="tab-list"] {{gap:26px;border-bottom:1px solid var(--line);}}
.stTabs [data-baseweb="tab"] {{color:#b8cbd4!important;}}
.stTabs [aria-selected="true"] {{color:white!important;border-bottom-color:var(--accent)!important;}}
div[data-testid="stMetric"] {{background:rgba(0,0,0,.10);border:1px solid var(--line);border-radius:12px;padding:12px 14px;}}
div[data-testid="stMetricLabel"] {{color:var(--muted)!important;}}
div[data-testid="stMetricValue"] {{color:white!important;font-size:1.42rem;}}
div[data-testid="stAlert"] {{background:rgba(0,0,0,.12)!important;color:white!important;border:1px solid var(--line)!important;border-radius:11px;}}
div[data-testid="stDataFrame"] {{border:1px solid var(--line);border-radius:12px;overflow:hidden;}}
hr {{border-color:var(--line)!important;}}

@media(max-width:1000px) {{
 .q-tag,.q-divider,.q-engine{{display:none}}
 .q-ready-grid{{grid-template-columns:1fr 1fr}}
 .block-container{{padding:4.2rem 1rem 2rem!important}}
 .q-hero h1{{font-size:32px}}
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
  <div class="q-pill">◆ Research Intelligence Workspace</div>
</div>
<div class="q-hero">
  <div>
    <h1>Research Transcriber</h1>
    <p>Accuracy-first Urdu + English interview transcription. Long recordings are divided into one-minute listening windows, checked for corruption, and reassembled into a research-ready record.</p>
  </div>
  <div class="q-engine">FREE ENGINE · WHISPER LARGE V3</div>
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
            f"{float(s.get('start', 0)):.3f}", f"{float(s.get('end', 0)):.3f}",
            format_seconds(float(s.get("start", 0))), format_seconds(float(s.get("end", 0))),
            str(s.get("text", "")).strip(), "YES" if s.get("review_flag") else "",
        ])
    return buf.getvalue()


def status_card(label: str, value: str, ok: bool = False, warn: bool = False) -> str:
    cls = "q-ok" if ok else "q-warn" if warn else ""
    return f'<div class="q-ready"><div class="q-ready-label">{html.escape(label)}</div><div class="q-ready-value {cls}">{html.escape(value)}</div></div>'


saved_groq = secret("GROQ_API_KEY")
if "session_groq_key" not in st.session_state:
    st.session_state.session_groq_key = ""

left, main = st.columns([0.25, 0.75], gap="large")

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

        st.markdown('<div class="q-panel-title" style="margin-top:20px">Outputs</div>', unsafe_allow_html=True)
        make_translation = st.toggle("English translation", value=True)

        with st.expander("Context & API", expanded=True):
            context = st.text_area(
                "Names / places / technical terms",
                placeholder="PASCO, HBL, Dreyfus, Layyah, names, acronyms…",
                height=90,
                max_chars=700,
            )
            if saved_groq:
                st.success("Groq connected from Streamlit Secrets")
            else:
                st.text_input("Groq API key", type="password", placeholder="gsk_…", key="groq_key_input")
                if st.button("Apply Groq key", use_container_width=True):
                    st.session_state.session_groq_key = str(st.session_state.get("groq_key_input", "") or "").strip()
                if st.session_state.session_groq_key:
                    st.success("Groq key active for this session")

    st.markdown(
        '<div class="q-note"><strong style="color:white">Research rule</strong><br>Silence is recorded as silence, not failure. Garbled sections stay flagged. Translation and summaries never hide failed source sections.</div>',
        unsafe_allow_html=True,
    )


groq_key = saved_groq or st.session_state.session_groq_key

with main:
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Drop interview audio or video",
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
        + status_card("Accuracy mode", "Large V3" if model == ACCURACY_MODEL else "Large V3 Turbo", ok=True)
        + status_card("English output", "Enabled" if make_translation else "Off", ok=make_translation)
        + '</div>'
    )
    st.markdown(readiness, unsafe_allow_html=True)

    if uploaded is not None:
        size_mb = uploaded.size / (1024 * 1024)
        a, b, c = st.columns(3)
        a.metric("Recording", uploaded.name)
        b.metric("Upload", f"{size_mb:.1f} MB")
        c.metric("Pipeline", "60 sec windows")

    action = "Run research transcription"
    start = st.button(
        action if ready else "Waiting for recording + Groq key",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    )

    pipeline_box = st.empty()
    live_box = st.empty()

    if not st.session_state.get("result") and not start:
        tabs = st.tabs(["Transcript", "Translation", "Quality", "Summary", "Downloads"])
        with tabs[0]: st.caption("The live draft appears here while the interview is being processed.")
        with tabs[1]: st.caption("English translation is generated only from accepted/reviewable source text.")
        with tabs[2]: st.caption("Speech, silence, review and failed windows are separated here.")
        with tabs[3]: st.caption("Generate an executive research summary after transcription.")
        with tabs[4]: st.caption("Download transcript, translation, timestamps, CSV and QA report here.")

    if start:
        st.session_state.pop("result", None)
        st.session_state.pop("research_summary", None)
        work_dir = Path(tempfile.mkdtemp(prefix="quantora_free_"))
        suffix = Path(uploaded.name).suffix.lower() or ".m4a"
        source_path = work_dir / f"source{suffix}"
        chunk_dir = work_dir / "chunks"
        progress = st.progress(0, text="Preparing recording…")
        live_chunks: list[dict] = []

        def pipeline(title: str, detail: str) -> None:
            pipeline_box.markdown(
                f'<div class="q-pipeline"><strong>{html.escape(title)}</strong><br><small>{html.escape(detail)}</small></div>',
                unsafe_allow_html=True,
            )

        def show_live() -> None:
            parts = []
            for c in live_chunks[-12:]:
                text = html.escape(str(c.get("text", "") or "")).replace("\n", "<br>")
                if not text:
                    text = "… no confident speech returned in this window"
                parts.append(
                    f'<div class="q-live-part"><div class="q-live-time">{format_seconds(float(c.get("keep_after", 0)))} — {format_seconds(float(c.get("end", 0)))}</div><div class="q-live-text">{text}</div></div>'
                )
            live_box.markdown(
                '<div class="q-live"><div class="q-live-title">Live transcript</div><div class="q-live-sub">Showing the latest listening windows. Final QA is applied after the speech pass.</div>' + "".join(parts) + '</div>',
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

            pipeline("Stage 2 · Accuracy preparation", "Creating short lossless listening windows. No speech model runs on the Streamlit CPU.")
            if duration <= 300 and source_path.stat().st_size <= MAX_CHUNK_BYTES:
                chunks = [{
                    "path": str(source_path), "duration": duration, "offset": 0.0,
                    "keep_after": 0.0, "nominal_end": duration, "overlap_seconds": 0.0,
                }]
            else:
                chunks = split_audio_lossless(str(source_path), str(chunk_dir))

            prompt = context.strip()
            if language_label == "Urdu + English (mixed)":
                prompt = (
                    "Pakistani Urdu-English research interview. اردو and English may switch naturally. "
                    "PASCO, procurement, private sector, agriculture, financing. " + prompt
                )[:700]

            def progress_cb(done: int, total: int, message: str) -> None:
                progress.progress(min(82, 5 + int((done / max(1, total)) * 77)), text=message)
                pipeline("Stage 3 · Speech recognition", f"{message} · {done}/{total} windows complete")

            def chunk_cb(chunk: dict) -> None:
                live_chunks.append(chunk)
                show_live()

            result = transcribe_chunks(
                chunks=chunks,
                api_key=groq_key,
                language=language_map[language_label],
                model=model,
                context_prompt=prompt,
                progress_callback=progress_cb,
                chunk_callback=chunk_cb,
            )

            pipeline("Stage 4 · Integrity scan", "Separating real speech from silence and flagging suspicious language output.")
            validated: list[dict] = []
            for cr in result.get("chunk_results", []):
                assessment = assess_single(
                    cr.get("text", ""),
                    api_error=cr.get("error"),
                    avg_no_speech_prob=cr.get("avg_no_speech_prob"),
                    max_no_speech_prob=cr.get("max_no_speech_prob"),
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

            progress.progress(88, text="Building final research outputs…")
            pipeline("Stage 5 · Research outputs", "Building validated transcript and optional English translation.")
            if make_translation:
                result["translation"] = translate_validated_chunks(
                    validated_chunks=validated,
                    api_key=groq_key,
                    progress_callback=None,
                )

            st.session_state["result"] = result
            progress.progress(100, text="Research transcript ready")
            pipeline("Complete ✓", "Transcript, quality map, translation and downloads are ready below.")

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
        f'''<div class="q-health"><div><div class="q-health-title">Transcript health · {html.escape(h_status)}</div><div class="q-health-sub">{health.get('passed',0)} speech windows passed · {health.get('review',0)} review · {health.get('failed',0)} failed · {health.get('silence',0)} silence</div></div><div class="q-score">{health.get('score',0)}/100</div></div>''',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration", format_seconds(duration))
    m2.metric("Listening windows", int(result.get("parts", 1)))
    m3.metric("Speech reliability", f"{health.get('score',0)}/100")
    m4.metric("Engine", "Whisper Large V3")

    tabs = st.tabs(["Transcript", "Translation", "Quality", "Summary", "Downloads"])

    with tabs[0]:
        for c in validated:
            state = c.get("status", "review")
            start = format_seconds(float(c.get("start", 0)))
            end = format_seconds(float(c.get("end", 0)))
            if state == "silence":
                st.markdown(
                    f'<div class="q-chunk"><div class="q-time">{start} — {end} · SILENCE</div><div class="q-silence">No confident speech detected.</div></div>',
                    unsafe_allow_html=True,
                )
                continue
            icon = "✓" if state == "passed" else "!" if state == "review" else "×"
            safe = html.escape(str(c.get("selected_text", "") or "")).replace("\n", "<br>")
            if not safe:
                safe = "[Transcription unavailable — listen back to this window.]"
            st.markdown(
                f'<div class="q-chunk"><div class="q-time">{start} — {end} · {icon} {state.upper()} · {c.get("score",0)}/100</div><div class="q-text">{safe}</div><div class="q-provider">Groq Whisper Large V3</div></div>',
                unsafe_allow_html=True,
            )
        with st.expander("Raw detailed timestamps"):
            st.text_area("Raw", value=timestamped_text(segments), height=520, label_visibility="collapsed")

    with tabs[1]:
        if result.get("translation"):
            st.text_area("English translation", value=translation_text(result["translation"]), height=650, label_visibility="collapsed")
            st.caption("FAILED source windows are withheld from translation rather than guessed.")
        else:
            st.info("English translation was disabled for this run.")

    with tabs[2]:
        rows = []
        for c in validated:
            rows.append({
                "time": f"{format_seconds(float(c.get('start',0)))}–{format_seconds(float(c.get('end',0)))}",
                "state": c.get("status"),
                "score": "—" if c.get("score") is None else c.get("score"),
                "no-speech": "" if c.get("avg_no_speech_prob") is None else f"{float(c.get('avg_no_speech_prob')):.2f}",
                "reason": ", ".join(c.get("reasons", [])),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("SILENCE is neutral and does not reduce the reliability score. REVIEW/FAILED windows are where you should listen back.")

    with tabs[3]:
        usable = [c for c in validated if c.get("status") in {"passed", "review"} and c.get("selected_text")]
        if "research_summary" not in st.session_state:
            st.caption("Summary is generated only from usable transcript windows; failed audio is excluded.")
            if st.button("Generate executive research summary", use_container_width=True):
                with st.spinner("Building summary…"):
                    st.session_state["research_summary"] = build_research_summary(validated_chunks=usable, api_key=groq_key)
                    st.rerun()
        else:
            summary = st.session_state["research_summary"]
            st.markdown("### English summary")
            st.markdown(summary.get("english", "") or "_No summary returned._")
            st.markdown("### Roman Urdu summary")
            st.markdown(summary.get("roman_urdu", "") or "_No Roman Urdu summary returned._")

    with tabs[4]:
        stem = Path(source_name).stem
        report = {
            "source_file": source_name,
            "health": health,
            "model": result.get("model"),
            "validated_chunks": validated,
        }
        st.download_button("Validated transcript · TXT", validated_text(validated).encode("utf-8"), f"{stem}_validated_transcript.txt", "text/plain", use_container_width=True)
        if result.get("translation"):
            st.download_button("English translation · TXT", translation_text(result["translation"]).encode("utf-8"), f"{stem}_english_translation.txt", "text/plain", use_container_width=True)
        st.download_button("Raw timestamped transcript · TXT", timestamped_text(segments).encode("utf-8"), f"{stem}_raw_timestamped.txt", "text/plain", use_container_width=True)
        st.download_button("Subtitles · SRT", transcript_to_srt(segments).encode("utf-8"), f"{stem}_raw.srt", "application/x-subrip", use_container_width=True)
        st.download_button("Research table · CSV", segments_to_csv(segments).encode("utf-8-sig"), f"{stem}_segments.csv", "text/csv", use_container_width=True)
        st.download_button("Quality report · JSON", json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"), f"{stem}_quality.json", "application/json", use_container_width=True)
