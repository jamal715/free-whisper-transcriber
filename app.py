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
:root{{--q:{PRIMARY};--ink:#111418;--muted:#66717c;--line:#dfe4e7;--soft:#f7f9fa;--ok:#18794e;--warn:#9a6700;}}
[data-testid="stSidebar"]{{display:none;}}
[data-testid="stHeader"]{{background:rgba(255,255,255,.98);height:3.25rem;}}
.stApp{{background:#fff;color:var(--ink);}}
.block-container{{max-width:1480px;padding:4.7rem 2.3rem 3rem!important;}}
html,body,[class*="css"]{{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;}}
.q-top{{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding:0 2px 17px;margin-bottom:24px;min-height:72px;}}
.q-brand{{display:flex;align-items:center;gap:14px;min-width:0;}}
.q-brand img{{width:58px!important;height:58px!important;object-fit:contain!important;object-position:center!important;border-radius:0!important;box-shadow:none!important;background:#fff!important;}}
.q-name{{font-size:29px;font-weight:800;letter-spacing:-.8px;color:#111418;}}
.q-divider{{width:1px;height:36px;background:#d9dfe3;margin:0 5px;}}
.q-tag{{font-size:14px;color:#52606b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:520px;}}
.q-pill{{border:1px solid #cad5db;border-radius:10px;padding:9px 13px;color:var(--q);background:#fbfcfd;font-size:13px;font-weight:700;}}
.q-hero{{text-align:center;padding:8px 0 20px;}}
.q-hero h1{{font-size:39px;letter-spacing:-1.3px;margin:.1rem 0 .55rem;line-height:1.1;}}
.q-hero p{{font-size:16px;color:#6a7480;margin:0;}}
.q-panel-title{{font-size:20px;font-weight:800;margin:2px 0 11px;}}
.q-security{{border:1px solid var(--line);border-radius:12px;padding:14px;background:#fafcfd;color:#5f6973;font-size:12px;line-height:1.55;margin-top:14px;}}
.q-connected{{color:var(--ok);font-size:12px;font-weight:750;margin-top:6px;}}
.q-readiness{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:12px 0 10px;}}
.q-ready{{border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:#fafcfd;font-size:12px;color:#5b6570;}}
.q-ready strong{{display:block;color:#161a1e;font-size:12px;margin-bottom:2px;}}
.q-ready.ok{{border-color:#b9d8ca;background:#f5fbf8;}}
.q-ready.warn{{border-color:#ead9a7;background:#fffbeb;}}
.q-run-note{{background:#f4f8fa;border:1px solid #d8e5ea;border-radius:11px;padding:12px 14px;color:#53616c;font-size:13px;margin:10px 0;}}
.q-live{{border:1px solid #cfe0e8;background:#fbfdfe;border-radius:12px;padding:14px 16px;margin-top:12px;}}
.q-live-title{{font-weight:800;color:var(--q);margin-bottom:8px;}}
.q-live-part{{padding:8px 0;border-top:1px solid #e7eef2;}}
.q-live-time{{font-size:11px;color:#687782;font-weight:700;}}
.q-live-text{{font-size:14px;line-height:1.6;color:#181c20;margin-top:3px;}}
.q-pipeline{{border:1px solid #d7e3e9;background:#f7fbfc;border-radius:12px;padding:12px 14px;margin:12px 0;}}
.q-pipeline strong{{color:var(--q);}}
.q-health{{border-left:4px solid var(--q);background:#f5fafc;padding:14px 16px;border-radius:9px;margin:8px 0 16px;}}
.q-health strong{{font-size:16px;}}
.q-chunk{{border-bottom:1px solid #e8ecef;padding:13px 2px 15px;}}
.q-time{{font-size:12px;color:var(--q);font-weight:750;}}
.q-text{{font-size:15px;line-height:1.72;color:#171a1d;margin-top:5px;}}
.q-provider{{font-size:11px;color:#8a939b;margin-top:5px;}}
div[data-testid="stVerticalBlockBorderWrapper"]{{border-color:var(--line)!important;border-radius:14px!important;box-shadow:0 1px 2px rgba(0,0,0,.02);}}
div[data-testid="stFileUploader"] section{{min-height:220px;border:1.5px dashed #a9c6d4;background:#fcfdfe;border-radius:14px;display:flex;align-items:center;}}
div[data-testid="stFileUploader"] button{{background:var(--q)!important;color:#fff!important;border:none!important;border-radius:8px!important;}}
.stButton>button[kind="primary"]{{background:var(--q)!important;color:#fff!important;border:1px solid var(--q)!important;min-height:50px;font-weight:750;border-radius:9px;}}
.stButton>button:disabled{{opacity:.52!important;}}
.stTabs [data-baseweb="tab-list"]{{gap:25px;border-bottom:1px solid var(--line);}}
.stTabs [aria-selected="true"]{{color:var(--q)!important;}}
div[data-testid="stMetricValue"]{{font-size:1.42rem;color:#111820;}}
div[data-testid="stAlert"]{{border-radius:10px;}}
@media(max-width:1000px){{.q-tag,.q-divider{{display:none}}.q-readiness{{grid-template-columns:1fr 1fr}}.block-container{{padding:4.2rem 1rem 2rem!important}}.q-hero h1{{font-size:31px}}}}
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
  <div class="q-pill">🔒 Research workspace</div>
</div>
<div class="q-hero">
  <h1>QuantOra Research Transcriber</h1>
  <p>Accurate transcription, validation and English translation for Urdu–English research interviews.</p>
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
        f"[{format_seconds(float(s.get('start',0)))} → {format_seconds(float(s.get('end',0)))}] {str(s.get('text','')).strip()}"
        for s in segments if str(s.get("text", "")).strip()
    )


def validated_text(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{format_seconds(float(c.get('start',0)))} → {format_seconds(float(c.get('end',0)))}]\n{str(c.get('selected_text','')).strip()}"
        for c in chunks if str(c.get("selected_text", "")).strip()
    )


def translation_text(result: dict) -> str:
    return "\n\n".join(
        f"[{format_seconds(float(c.get('start',0)))} → {format_seconds(float(c.get('end',0)))}]\n{str(c.get('text','')).strip()}"
        for c in result.get("chunks", []) if str(c.get("text", "")).strip()
    )


def segments_to_csv(segments: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["start_seconds", "end_seconds", "start", "end", "text", "review_flag"])
    for s in segments:
        w.writerow([
            f"{float(s.get('start',0)):.3f}",
            f"{float(s.get('end',0)):.3f}",
            format_seconds(float(s.get("start",0))),
            format_seconds(float(s.get("end",0))),
            str(s.get("text", "")).strip(),
            "YES" if s.get("review_flag") else "",
        ])
    return buf.getvalue()


def render_readiness(file_ready: bool, groq_ready: bool, openai_ready: bool, translation_on: bool) -> str:
    def card(title: str, value: str, state: str) -> str:
        return f'<div class="q-ready {state}"><strong>{html.escape(title)}</strong>{html.escape(value)}</div>'
    return (
        '<div class="q-readiness">'
        + card("Recording", "Ready ✓" if file_ready else "Waiting", "ok" if file_ready else "")
        + card("Groq transcription", "Ready ✓" if groq_ready else "API key needed", "ok" if groq_ready else "warn")
        + card("Independent validation", "Ready ✓" if openai_ready else "Optional / not connected", "ok" if openai_ready else "")
        + card("English translation", "On ✓" if translation_on else "Off", "ok" if translation_on else "")
        + '</div>'
    )


saved_groq = secret("GROQ_API_KEY")
saved_openai = secret("OPENAI_API_KEY")
if "session_groq_key" not in st.session_state:
    st.session_state.session_groq_key = ""
if "session_openai_key" not in st.session_state:
    st.session_state.session_openai_key = ""

left, main = st.columns([0.23, 0.77], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="q-panel-title">Language</div>', unsafe_allow_html=True)
        language_label = st.selectbox(
            "Language",
            ["Urdu + English (mixed)", "Mostly Urdu", "Mostly English"],
            index=0,
            label_visibility="collapsed",
            help="For normal Pakistani code-switching, keep Mixed.",
        )
        language_map = {"Urdu + English (mixed)": None, "Mostly Urdu": "ur", "Mostly English": "en"}

        st.markdown('<div class="q-panel-title" style="margin-top:20px">Transcription mode</div>', unsafe_allow_html=True)
        mode_label = st.selectbox(
            "Mode",
            ["Maximum confidence", "Research accuracy", "Fast draft"],
            index=0,
            label_visibility="collapsed",
            help="Maximum confidence = Groq Large V3 + independent OpenAI validation when configured.",
        )
        model = FAST_MODEL if mode_label == "Fast draft" else ACCURACY_MODEL

        st.markdown('<div class="q-panel-title" style="margin-top:20px">Translation</div>', unsafe_allow_html=True)
        make_translation = st.toggle(
            "Create English translation",
            value=True,
            help="Recommended for mixed Urdu/English interviews. Translation is made from the validated transcript.",
        )

        with st.expander("Advanced", expanded=True):
            context = st.text_area(
                "Names / places / research terms",
                placeholder="HBL, PASCO, IFC, people, places, technical terms…",
                height=90,
                max_chars=900,
            )

            if saved_groq:
                st.markdown('<div class="q-connected">● Groq connected from Streamlit Secrets</div>', unsafe_allow_html=True)
            else:
                st.text_input(
                    "Groq API key",
                    type="password",
                    placeholder="gsk_…",
                    key="groq_key_input",
                    help="Paste the key, then press Apply API keys below.",
                )

            if saved_openai:
                st.markdown('<div class="q-connected">● OpenAI validator connected from Streamlit Secrets</div>', unsafe_allow_html=True)
            else:
                st.text_input(
                    "OpenAI API key (optional)",
                    type="password",
                    placeholder="sk-…",
                    key="openai_key_input",
                    help="Optional second speech engine for independent validation/rescue.",
                )

            if not saved_groq or not saved_openai:
                if st.button("Apply API keys for this session", use_container_width=True, key="apply_keys"):
                    if not saved_groq:
                        st.session_state.session_groq_key = str(st.session_state.get("groq_key_input", "") or "").strip()
                    if not saved_openai:
                        st.session_state.session_openai_key = str(st.session_state.get("openai_key_input", "") or "").strip()

                if st.session_state.session_groq_key:
                    st.markdown('<div class="q-connected">● Groq key applied for this session</div>', unsafe_allow_html=True)
                if st.session_state.session_openai_key:
                    st.markdown('<div class="q-connected">● OpenAI key applied for this session</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="q-security"><strong>Research privacy</strong><br>Temporary audio/chunks are deleted after each run. Store API keys in Streamlit Secrets for the permanent setup. Do not refresh the page while a long interview is processing.</div>',
        unsafe_allow_html=True,
    )


groq_key = saved_groq or st.session_state.session_groq_key
openai_key = saved_openai or st.session_state.session_openai_key

with main:
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Drag & drop your audio or video file here",
            type=["m4a", "mp3", "mp4", "wav", "flac", "ogg", "webm", "mpeg", "mpga"],
            accept_multiple_files=False,
            help="Designed for interviews up to 120 minutes. MP3/M4A is recommended for long interviews.",
        )

    file_ready = uploaded is not None
    groq_ready = bool(groq_key)
    openai_ready = bool(openai_key)
    ready = file_ready and groq_ready

    st.markdown(render_readiness(file_ready, groq_ready, openai_ready, make_translation), unsafe_allow_html=True)

    if uploaded is not None:
        size_mb = uploaded.size / (1024 * 1024)
        a, b, c = st.columns(3)
        a.metric("File", uploaded.name)
        b.metric("Size", f"{size_mb:.1f} MB")
        if mode_label == "Maximum confidence" and openai_ready:
            c.metric("Validation", "Dual engine")
        elif mode_label == "Maximum confidence":
            c.metric("Validation", "Integrity scan")
        else:
            c.metric("Model", "Large V3" if model == ACCURACY_MODEL else "Large V3 Turbo")

        st.markdown(
            '<div class="q-run-note">Long recordings are automatically split into safe overlapping parts. A live draft appears below as each part finishes. The final transcript can change after independent validation/rescue.</div>',
            unsafe_allow_html=True,
        )

    if not groq_ready:
        st.info("Open Advanced, paste your Groq key, then click **Apply API keys for this session**.")
    if mode_label == "Maximum confidence" and not openai_ready:
        st.warning("You can run now with Groq + integrity checks. Add an OpenAI API key for independent chunk-by-chunk validation and rescue.")

    action = "Start transcription + validation" + (" + translation" if make_translation else "")
    start = st.button(
        action if ready else "Waiting for recording and Groq key…",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    )

    pipeline_box = st.empty()
    live_box = st.empty()

    if not st.session_state.get("result") and not start:
        tabs = st.tabs(["Transcript", "Translation", "Validation", "Summary", "Downloads"])
        with tabs[0]:
            st.info("Your live transcript will appear here once you press **Start transcription**. Final validated text replaces the live draft when the run finishes.")
        with tabs[1]:
            st.caption("The English translation appears here after transcription and validation.")
        with tabs[2]:
            st.caption("Chunk-by-chunk PASS / REVIEW / FAILED checks appear here after validation.")
        with tabs[3]:
            st.caption("A research summary can be generated from the validated transcript after the run.")
        with tabs[4]:
            st.caption("Validated transcript, translation, raw timestamps and validation report become downloadable here.")

    if start:
        st.session_state.pop("result", None)
        st.session_state.pop("research_summary", None)
        work_dir = Path(tempfile.mkdtemp(prefix="quantora_research_"))
        source_path = work_dir / f"source{Path(uploaded.name).suffix.lower() or '.m4a'}"
        chunk_dir = work_dir / "chunks"
        progress = st.progress(0, text="Receiving interview…")
        status = st.status("QuantOra processing pipeline", expanded=True)
        live_chunks: list[dict] = []

        def show_pipeline(step: int, detail: str) -> None:
            pipeline_box.markdown(
                f'<div class="q-pipeline"><strong>Processing is active — Step {step}/5</strong><br>{html.escape(detail)}<br><span style="font-size:12px;color:#71808a">Keep this browser tab open. Do not refresh while the run is active.</span></div>',
                unsafe_allow_html=True,
            )

        def show_live() -> None:
            parts = []
            for c in live_chunks:
                text = html.escape(str(c.get("text", "") or "")).replace("\n", "<br>")
                if not text:
                    text = "[No usable draft text returned for this part — validation may rescue it.]"
                parts.append(
                    f'<div class="q-live-part"><div class="q-live-time">Part {c.get("index")} · {format_seconds(float(c.get("keep_after",0)))} — {format_seconds(float(c.get("end",0)))}</div><div class="q-live-text">{text}</div></div>'
                )
            live_box.markdown(
                '<div class="q-live"><div class="q-live-title">Live draft transcript</div><div style="font-size:12px;color:#697782;margin-bottom:6px">This updates as each audio part finishes. It is not final until validation completes.</div>'
                + "".join(parts)
                + '</div>',
                unsafe_allow_html=True,
            )

        try:
            show_pipeline(1, "Receiving the uploaded recording and checking its duration.")
            with source_path.open("wb") as out:
                uploaded.seek(0)
                shutil.copyfileobj(uploaded, out, length=1024 * 1024)

            duration = probe_duration(str(source_path))
            if not duration:
                raise RuntimeError("Could not read recording duration. For long interviews, use MP3 or M4A.")
            if duration > MAX_DURATION_SECONDS + 0.5:
                raise RuntimeError(f"This workspace supports up to 120 minutes. Your recording is {format_seconds(duration)}.")
            status.write(f"1/5 Audio ready — {format_seconds(duration)}")

            show_pipeline(2, "Preparing safe audio parts. Large files are split automatically without re-encoding.")
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
                progress.progress(5, text="Preparing safe overlapping chunks…")
                chunks = split_audio_lossless(str(source_path), str(chunk_dir))
            status.write(f"2/5 Prepared {len(chunks)} safe part{'s' if len(chunks) != 1 else ''}")

            mixed_prompt = ""
            if language_label == "Urdu + English (mixed)":
                mixed_prompt = (
                    "Pakistani research interview with natural Urdu-English code-switching. "
                    "Write Urdu in Urdu Perso-Arabic script and English in Latin script. "
                    "Never output Devanagari/Hindi unless genuinely spoken. Preserve names, numbers, acronyms and technical terms exactly. "
                )
            speech_context = (mixed_prompt + context.strip())[:850]

            def tp(done: int, total: int, message: str) -> None:
                ceiling = 43 if (mode_label == "Maximum confidence" and openai_ready) else 70
                progress.progress(min(76, 7 + int((done / max(1, total)) * ceiling)), text=message)
                show_pipeline(3, message)
                status.write("3/5 " + message)

            def on_chunk(chunk_result: dict) -> None:
                live_chunks.append(chunk_result)
                show_live()

            result = transcribe_chunks(
                chunks=chunks,
                api_key=groq_key,
                language=language_map[language_label],
                model=model,
                context_prompt=speech_context,
                progress_callback=tp,
                chunk_callback=on_chunk,
            )

            validated: list[dict] = []
            verifier_model = None
            chunk_results = result.get("chunk_results", [])
            show_pipeline(4, "Checking each transcript part for corruption and, when configured, comparing it with an independent OpenAI transcription.")

            for idx, cr in enumerate(chunk_results):
                verifier_text = None
                verifier_error = None
                if mode_label == "Maximum confidence" and openai_ready:
                    progress.progress(
                        52 + int((idx / max(1, len(chunk_results))) * 27),
                        text=f"Validating part {idx + 1} of {len(chunk_results)}…",
                    )
                    status.write(f"4/5 Independent validation — part {idx + 1} of {len(chunk_results)}")
                    try:
                        vr = transcribe_for_validation(
                            path=chunks[idx]["path"],
                            api_key=openai_key,
                            context_prompt=(
                                "Pakistani research interview; Urdu-English code-switching is expected. Preserve wording. "
                                "Urdu uses Perso-Arabic script, English remains Latin. Never invent missing speech. "
                                + context.strip()
                            )[:900],
                        )
                        verifier_text = vr["text"]
                        verifier_model = vr.get("model")
                    except Exception as exc:
                        verifier_error = str(exc)

                assessment = assess_dual(cr.get("text", ""), verifier_text)
                if cr.get("error"):
                    assessment["reasons"] = list(assessment.get("reasons", [])) + ["primary engine error: " + str(cr["error"])[:180]]
                    if verifier_text:
                        assessment["selected_text"] = verifier_text
                        assessment["selected_provider"] = "OpenAI rescue after primary failure"
                        assessment["status"] = "review"
                        assessment["score"] = max(68, int(assessment.get("score", 0)))
                    else:
                        assessment["status"] = "failed"
                        assessment["score"] = 0

                assessment.update({
                    "index": idx + 1,
                    "start": float(cr.get("keep_after", cr.get("offset", 0.0))),
                    "end": float(cr.get("end", 0.0)),
                    "groq_text": cr.get("text", ""),
                    "verifier_text": verifier_text,
                    "verifier_error": verifier_error,
                })

                if verifier_error:
                    assessment["reasons"] = list(assessment.get("reasons", [])) + ["independent validator unavailable for this part"]
                    if assessment["status"] == "passed":
                        assessment["status"] = "review"
                        assessment["score"] = min(int(assessment["score"]), 78)
                validated.append(assessment)

            health = overall_health(validated)
            result["validated_chunks"] = validated
            result["health"] = health
            result["verifier_model"] = verifier_model
            result["validated_text"] = "\n\n".join(c.get("selected_text", "").strip() for c in validated if c.get("selected_text", "").strip())

            show_pipeline(5, "Building the final validated transcript" + (" and English translation." if make_translation else "."))
            if make_translation:
                def trp(done: int, total: int, message: str) -> None:
                    progress.progress(min(97, 80 + int((done / max(1, total)) * 17)), text=message)
                    status.write("5/5 " + message)

                result["translation"] = translate_validated_chunks(
                    validated_chunks=validated,
                    api_key=groq_key,
                    progress_callback=trp,
                )

            result["source_name"] = uploaded.name
            st.session_state["result"] = result
            progress.progress(100, text="Interview ready")
            pipeline_box.markdown(
                '<div class="q-pipeline"><strong>Processing complete ✓</strong><br>The final validated transcript is below. Use the Validation tab to review flagged time windows.</div>',
                unsafe_allow_html=True,
            )
            status.update(label="Processing complete", state="complete", expanded=False)

        except Exception as exc:
            pipeline_box.markdown(
                '<div class="q-pipeline"><strong>Processing stopped</strong><br>' + html.escape(str(exc)) + '</div>',
                unsafe_allow_html=True,
            )
            status.update(label="Processing failed", state="error", expanded=True)
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
    st.markdown(
        f'<div class="q-health"><strong>Transcript health: {html.escape(str(health.get("status", "unknown")).title())}</strong> · Score {health.get("score",0)}/100 · {health.get("passed",0)} passed · {health.get("review",0)} review · {health.get("failed",0)} failed</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration", format_seconds(duration))
    m2.metric("Parts", int(result.get("parts", 1)))
    m3.metric("Reliability", f"{health.get('score',0)}/100")
    m4.metric("Validation", "Dual engine" if result.get("verifier_model") else "Single engine + QA")

    tabs = st.tabs(["Transcript", "Translation", "Validation", "Summary", "Downloads"])

    with tabs[0]:
        for c in validated:
            state = c.get("status", "review")
            icon = "✓" if state == "passed" else "!" if state == "review" else "×"
            safe = html.escape(str(c.get("selected_text", ""))).replace("\n", "<br>")
            st.markdown(
                f'<div class="q-chunk"><div class="q-time">{format_seconds(float(c.get("start",0)))} — {format_seconds(float(c.get("end",0)))} &nbsp; {icon} {state.upper()} · {c.get("score",0)}/100</div><div class="q-text">{safe}</div><div class="q-provider">{html.escape(str(c.get("selected_provider","Groq")))}</div></div>',
                unsafe_allow_html=True,
            )
        with st.expander("Raw detailed timestamps"):
            st.text_area("Raw", value=timestamped_text(segments), height=520, label_visibility="collapsed")

    with tabs[1]:
        if result.get("translation"):
            st.text_area("English translation", value=translation_text(result["translation"]), height=650, label_visibility="collapsed")
        else:
            st.info("Translation was not selected for this run.")

    with tabs[2]:
        rows = []
        for c in validated:
            sim = c.get("similarity")
            rows.append({
                "time": f"{format_seconds(float(c.get('start',0)))}–{format_seconds(float(c.get('end',0)))}",
                "status": c.get("status"),
                "score": c.get("score"),
                "engine agreement": "" if sim is None else f"{float(sim):.0%}",
                "selected": c.get("selected_provider"),
                "reason": ", ".join(c.get("reasons", [])),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("Listen back to REVIEW/FAILED windows and verify any direct quotation before publication.")

    with tabs[3]:
        if "research_summary" not in st.session_state:
            st.caption("Generate a clean English summary plus Roman Urdu from the validated transcript.")
            if st.button("Generate research summary", use_container_width=True, key="summary_btn"):
                with st.spinner("Building research summary…"):
                    st.session_state["research_summary"] = build_research_summary(validated_chunks=validated, api_key=groq_key)
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
            "verifier_model": result.get("verifier_model"),
            "validated_chunks": validated,
        }
        st.download_button("Download validated transcript (.txt)", validated_text(validated).encode("utf-8"), f"{stem}_validated_transcript.txt", "text/plain", use_container_width=True)
        if result.get("translation"):
            st.download_button("Download English translation (.txt)", translation_text(result["translation"]).encode("utf-8"), f"{stem}_english_translation.txt", "text/plain", use_container_width=True)
        st.download_button("Download raw timestamped transcript (.txt)", timestamped_text(segments).encode("utf-8"), f"{stem}_raw_timestamped.txt", "text/plain", use_container_width=True)
        st.download_button("Download raw subtitles (.srt)", transcript_to_srt(segments).encode("utf-8"), f"{stem}_raw.srt", "application/x-subrip", use_container_width=True)
        st.download_button("Download research table (.csv)", segments_to_csv(segments).encode("utf-8-sig"), f"{stem}_segments.csv", "text/csv", use_container_width=True)
        st.download_button("Download validation report (.json)", json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"), f"{stem}_validation.json", "application/json", use_container_width=True)
