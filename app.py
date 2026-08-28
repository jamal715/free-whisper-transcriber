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
:root {{ --q:{PRIMARY}; --ink:#111418; --muted:#66717c; --line:#dfe4e7; --soft:#f7f9fa; }}
[data-testid="stSidebar"] {{display:none;}}
[data-testid="stHeader"] {{background:rgba(255,255,255,.97);}}
.stApp {{background:#fff;color:var(--ink);}}
.block-container {{max-width:1480px;padding:1.0rem 2.3rem 3rem;}}
html,body,[class*="css"] {{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;}}
.q-top {{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding:0 2px 17px;margin-bottom:30px;}}
.q-brand {{display:flex;align-items:center;gap:14px;min-width:0;}}
.q-brand img {{width:66px!important;height:66px!important;object-fit:contain!important;border-radius:0!important;box-shadow:none!important;background:transparent!important;}}
.q-name {{font-size:30px;font-weight:800;letter-spacing:-.8px;color:#111418;}}
.q-divider {{width:1px;height:38px;background:#d9dfe3;margin:0 5px;}}
.q-tag {{font-size:14px;color:#52606b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:520px;}}
.q-pill {{border:1px solid #cad5db;border-radius:10px;padding:9px 13px;color:var(--q);background:#fbfcfd;font-size:13px;font-weight:700;}}
.q-hero {{text-align:center;padding:6px 0 22px;}}
.q-hero h1 {{font-size:41px;letter-spacing:-1.5px;margin:.2rem 0 .55rem;line-height:1.08;}}
.q-hero p {{font-size:17px;color:#6a7480;margin:0;}}
.q-panel-title {{font-size:21px;font-weight:800;margin:2px 0 11px;}}
.q-security {{border:1px solid var(--line);border-radius:12px;padding:14px;background:#fafcfd;color:#5f6973;font-size:12px;line-height:1.55;margin-top:14px;}}
.q-connected {{color:var(--q);font-size:12px;font-weight:700;margin-top:6px;}}
.q-run-note {{background:#f4f8fa;border:1px solid #d8e5ea;border-radius:11px;padding:12px 14px;color:#53616c;font-size:13px;margin:10px 0;}}
.q-health {{border-left:4px solid var(--q);background:#f5fafc;padding:14px 16px;border-radius:9px;margin:8px 0 16px;}}
.q-health strong {{font-size:16px;}}
.q-chunk {{border-bottom:1px solid #e8ecef;padding:13px 2px 15px;}}
.q-time {{font-size:12px;color:var(--q);font-weight:750;}}
.q-text {{font-size:15px;line-height:1.72;color:#171a1d;margin-top:5px;}}
.q-provider {{font-size:11px;color:#8a939b;margin-top:5px;}}
div[data-testid="stVerticalBlockBorderWrapper"] {{border-color:var(--line)!important;border-radius:14px!important;box-shadow:0 1px 2px rgba(0,0,0,.02);}}
div[data-testid="stFileUploader"] section {{min-height:235px;border:1.5px dashed #a9c6d4;background:#fcfdfe;border-radius:14px;display:flex;align-items:center;}}
div[data-testid="stFileUploader"] button {{background:var(--q)!important;color:#fff!important;border:none!important;border-radius:8px!important;}}
.stButton>button[kind="primary"] {{background:var(--q)!important;color:#fff!important;border:1px solid var(--q)!important;min-height:50px;font-weight:750;border-radius:9px;}}
.stButton>button:disabled {{opacity:.55!important;}}
.stTabs [data-baseweb="tab-list"] {{gap:25px;border-bottom:1px solid var(--line);}}
.stTabs [aria-selected="true"] {{color:var(--q)!important;}}
div[data-testid="stMetricValue"] {{font-size:1.42rem;color:#111820;}}
div[data-testid="stAlert"] {{border-radius:10px;}}
@media(max-width:900px) {{.q-tag,.q-divider{{display:none}} .block-container{{padding:1rem}} .q-hero h1{{font-size:31px}}}}
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
            f"{float(s.get('start',0)):.3f}", f"{float(s.get('end',0)):.3f}",
            format_seconds(float(s.get("start",0))), format_seconds(float(s.get("end",0))),
            str(s.get("text", "")).strip(), "YES" if s.get("review_flag") else "",
        ])
    return buf.getvalue()

saved_groq = secret("GROQ_API_KEY")
saved_openai = secret("OPENAI_API_KEY")

left, main = st.columns([0.23, 0.77], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="q-panel-title">Language</div>', unsafe_allow_html=True)
        language_label = st.selectbox(
            "Language", ["Urdu + English (mixed)", "Mostly Urdu", "Mostly English"],
            index=0, label_visibility="collapsed",
            help="For normal Pakistani code-switching, keep Mixed.",
        )
        language_map = {"Urdu + English (mixed)":None, "Mostly Urdu":"ur", "Mostly English":"en"}

        st.markdown('<div class="q-panel-title" style="margin-top:20px">Transcription mode</div>', unsafe_allow_html=True)
        mode_label = st.selectbox(
            "Mode", ["Maximum confidence", "Research accuracy", "Fast draft"],
            index=0, label_visibility="collapsed",
            help="Maximum confidence = Groq Large V3 + independent OpenAI validation when configured.",
        )
        model = FAST_MODEL if mode_label == "Fast draft" else ACCURACY_MODEL

        st.markdown('<div class="q-panel-title" style="margin-top:20px">Translation</div>', unsafe_allow_html=True)
        make_translation = st.toggle(
            "Create English translation", value=True,
            help="Recommended for mixed Urdu/English interviews. Translation is made from the validated transcript.",
        )

        with st.expander("Advanced", expanded=False):
            context = st.text_area(
                "Names / places / research terms",
                placeholder="HBL, PASCO, IFC, people, places, technical terms…",
                height=95, max_chars=900,
            )
            groq_key = saved_groq or st.text_input("Groq API key", type="password", placeholder="gsk_…").strip()
            openai_key = saved_openai or st.text_input(
                "OpenAI API key (optional)", type="password", placeholder="sk-…",
                help="Used only for independent validation/rescue in Maximum confidence mode.",
            ).strip()
            if saved_groq:
                st.markdown('<div class="q-connected">● Groq connected from Secrets</div>', unsafe_allow_html=True)
            if saved_openai:
                st.markdown('<div class="q-connected">● OpenAI validator connected from Secrets</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="q-security"><strong>Research privacy</strong><br>Temporary audio/chunks are deleted after the run. Store API keys in Streamlit Secrets. For sensitive interviews, enable provider zero-data-retention controls where available.</div>',
        unsafe_allow_html=True,
    )

with main:
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Drag & drop your audio or video file here",
            type=["m4a","mp3","mp4","wav","flac","ogg","webm","mpeg","mpga"],
            accept_multiple_files=False,
            help="Designed for interviews up to 120 minutes. MP3/M4A is recommended for long interviews.",
        )

    ready = uploaded is not None and bool(groq_key)

    if uploaded is not None:
        size_mb = uploaded.size / (1024*1024)
        a,b,c = st.columns(3)
        a.metric("File", uploaded.name)
        b.metric("Size", f"{size_mb:.1f} MB")
        if mode_label == "Maximum confidence" and openai_key:
            c.metric("Validation", "Dual engine")
        elif mode_label == "Maximum confidence":
            c.metric("Validation", "Integrity scan")
        else:
            c.metric("Model", "Large V3" if model == ACCURACY_MODEL else "Large V3 Turbo")
        st.markdown(
            '<div class="q-run-note">Long recordings are automatically split into safe overlapping parts, transcribed sequentially, checked for corruption, stitched back together, then translated from the validated transcript.</div>',
            unsafe_allow_html=True,
        )

    if not groq_key:
        st.info("Add your Groq API key under Advanced to enable transcription.")
    if mode_label == "Maximum confidence" and not openai_key:
        st.warning("Maximum confidence is currently single-engine + integrity checks. Add an OpenAI API key under Advanced for independent chunk-by-chunk validation and rescue.")

    action = "Transcribe + validate" + (" + translate" if make_translation else "")
    start = st.button(action if ready else "Waiting for file and API key…", type="primary", use_container_width=True, disabled=not ready)

    if start:
        st.session_state.pop("result", None)
        st.session_state.pop("research_summary", None)
        work_dir = Path(tempfile.mkdtemp(prefix="quantora_research_"))
        source_path = work_dir / f"source{Path(uploaded.name).suffix.lower() or '.m4a'}"
        chunk_dir = work_dir / "chunks"
        progress = st.progress(0, text="Receiving interview…")
        status = st.status("QuantOra processing pipeline", expanded=True)

        try:
            with source_path.open("wb") as out:
                uploaded.seek(0)
                shutil.copyfileobj(uploaded, out, length=1024*1024)

            duration = probe_duration(str(source_path))
            if not duration:
                raise RuntimeError("Could not read recording duration. For long interviews, use MP3 or M4A.")
            if duration > MAX_DURATION_SECONDS + .5:
                raise RuntimeError(f"This workspace supports up to 120 minutes. Your recording is {format_seconds(duration)}.")
            status.write(f"1/5 Audio ready — {format_seconds(duration)}")

            if source_path.stat().st_size <= MAX_CHUNK_BYTES:
                chunks = [{"path":str(source_path),"duration":duration,"offset":0.0,"keep_after":0.0,"nominal_end":duration,"overlap_seconds":0.0}]
            else:
                progress.progress(5, text="Preparing safe overlapping chunks…")
                chunks = split_audio_lossless(str(source_path), str(chunk_dir))
            status.write(f"2/5 Prepared {len(chunks)} safe part{'s' if len(chunks)!=1 else ''}")

            mixed_prompt = ""
            if language_label == "Urdu + English (mixed)":
                mixed_prompt = (
                    "Pakistani research interview with natural Urdu-English code-switching. "
                    "Write Urdu in Urdu Perso-Arabic script and English in Latin script. "
                    "Never output Devanagari/Hindi unless genuinely spoken. Preserve names, numbers, acronyms and technical terms exactly. "
                )
            speech_context = (mixed_prompt + context.strip())[:850]

            def tp(done:int,total:int,message:str):
                ceiling = 43 if (mode_label=="Maximum confidence" and openai_key) else 70
                progress.progress(min(76, 7 + int((done/max(1,total))*ceiling)), text=message)
                status.write("3/5 " + message)

            result = transcribe_chunks(
                chunks=chunks, api_key=groq_key, language=language_map[language_label],
                model=model, context_prompt=speech_context, progress_callback=tp,
            )

            validated=[]
            verifier_model=None
            chunk_results=result.get("chunk_results", [])
            for idx, cr in enumerate(chunk_results):
                verifier_text=None; verifier_error=None
                if mode_label=="Maximum confidence" and openai_key:
                    progress.progress(52 + int((idx/max(1,len(chunk_results)))*27), text=f"Validating part {idx+1} of {len(chunk_results)}…")
                    status.write(f"4/5 Independent validation — part {idx+1} of {len(chunk_results)}")
                    try:
                        vr=transcribe_for_validation(
                            path=chunks[idx]["path"], api_key=openai_key,
                            context_prompt=(
                                "Pakistani research interview; Urdu-English code-switching is expected. Preserve wording. "
                                "Urdu uses Perso-Arabic script, English remains Latin. Never invent missing speech. " + context.strip()
                            )[:900],
                        )
                        verifier_text=vr["text"]; verifier_model=vr.get("model")
                    except Exception as exc:
                        verifier_error=str(exc)

                assessment=assess_dual(cr.get("text", ""), verifier_text)
                assessment.update({
                    "index":idx+1,
                    "start":float(cr.get("keep_after", cr.get("offset",0.0))),
                    "end":float(cr.get("end",0.0)),
                    "groq_text":cr.get("text", ""),
                    "verifier_text":verifier_text,
                    "verifier_error":verifier_error,
                })
                if verifier_error and assessment.get("status")=="passed":
                    assessment["status"]="review"; assessment["score"]=min(int(assessment.get("score",0)),78)
                    assessment["reasons"]=list(assessment.get("reasons",[]))+["independent validator unavailable"]
                validated.append(assessment)

            health=overall_health(validated)
            result.update({"validated_chunks":validated,"health":health,"verifier_model":verifier_model,"source_name":uploaded.name,"requested_model":model})
            result["validated_text"]="\n\n".join(c.get("selected_text","").strip() for c in validated if c.get("selected_text","").strip())

            if make_translation:
                status.write("5/5 Translating the validated transcript to English")
                def trp(done:int,total:int,message:str):
                    progress.progress(min(98,80+int((done/max(1,total))*18)), text=message)
                result["translation"]=translate_validated_chunks(validated_chunks=validated, api_key=groq_key, progress_callback=trp)

            st.session_state["result"]=result
            progress.progress(100, text="Interview ready")
            final_label="Completed — high confidence"
            if health.get("failed",0): final_label="Completed — failed sections require review"
            elif health.get("review",0): final_label="Completed — review flagged sections"
            status.update(label=final_label, state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Processing failed", state="error", expanded=True)
            st.error(str(exc))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

result=st.session_state.get("result")
if result:
    validated=result.get("validated_chunks",[]); health=result.get("health",{}); segments=result.get("segments",[])
    duration=float(result.get("duration",0.0) or 0.0); source_name=result.get("source_name","interview")
    st.markdown("---")
    st.markdown(
        f'<div class="q-health"><strong>Transcript health: {html.escape(str(health.get("status","unknown")).title())}</strong> &nbsp;·&nbsp; Score {health.get("score",0)}/100 &nbsp;·&nbsp; {health.get("passed",0)} passed &nbsp;·&nbsp; {health.get("review",0)} review &nbsp;·&nbsp; {health.get("failed",0)} failed</div>',
        unsafe_allow_html=True,
    )
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Duration",format_seconds(duration)); m2.metric("Parts",int(result.get("parts",1))); m3.metric("Reliability",f'{health.get("score",0)}/100'); m4.metric("Validation","Dual engine" if result.get("verifier_model") else "Single engine + QA")

    tabs=st.tabs(["Transcript","English translation","Validation","Summary","Downloads"])
    with tabs[0]:
        for c in validated:
            state=c.get("status","review"); icon="✓" if state=="passed" else "!" if state=="review" else "×"
            safe=html.escape(str(c.get("selected_text", ""))).replace("\n","<br>")
            st.markdown(f'<div class="q-chunk"><div class="q-time">{format_seconds(float(c.get("start",0)))} — {format_seconds(float(c.get("end",0)))} &nbsp; {icon} {state.upper()} · {c.get("score",0)}/100</div><div class="q-text">{safe}</div><div class="q-provider">{html.escape(str(c.get("selected_provider","Groq")))}</div></div>', unsafe_allow_html=True)
        with st.expander("Raw detailed timestamps"):
            st.text_area("Raw",value=timestamped_text(segments),height=500,label_visibility="collapsed")

    with tabs[1]:
        if result.get("translation"):
            st.text_area("Translation",value=translation_text(result["translation"]),height=650,label_visibility="collapsed")
            st.caption("English translation is generated from the validated transcript, not by retranscribing the full audio.")
        else:
            st.info("Translation was not selected for this run.")

    with tabs[2]:
        rows=[]
        for c in validated:
            sim=c.get("similarity")
            rows.append({"time":f'{format_seconds(float(c.get("start",0)))}–{format_seconds(float(c.get("end",0)))}',"status":c.get("status"),"score":c.get("score"),"engine agreement":"" if sim is None else f'{float(sim):.0%}',"selected":c.get("selected_provider"),"reason":", ".join(c.get("reasons",[]))})
        st.dataframe(rows,use_container_width=True,hide_index=True)
        st.caption("Direct quotations for publication should still be checked against the original audio at the indicated timestamp.")

    with tabs[3]:
        if "research_summary" not in st.session_state:
            st.caption("Generate a clean English research summary plus a Roman Urdu version from the validated transcript.")
            if st.button("Generate research summary",use_container_width=True):
                with st.spinner("Building summary…"):
                    try:
                        st.session_state["research_summary"]=build_research_summary(validated_chunks=validated,api_key=groq_key)
                        st.rerun()
                    except Exception as exc: st.error(str(exc))
        else:
            s=st.session_state["research_summary"]
            st.markdown("### English summary"); st.markdown(s.get("english","") or "_No summary returned._")
            st.markdown("### Roman Urdu summary"); st.markdown(s.get("roman_urdu","") or "_No Roman Urdu summary returned._")

    with tabs[4]:
        stem=Path(source_name).stem
        st.download_button("Download validated transcript (.txt)",validated_text(validated).encode("utf-8"),f"{stem}_validated_transcript.txt","text/plain",use_container_width=True)
        if result.get("translation"):
            st.download_button("Download English translation (.txt)",translation_text(result["translation"]).encode("utf-8"),f"{stem}_english_translation.txt","text/plain",use_container_width=True)
        st.download_button("Download raw timestamped transcript (.txt)",timestamped_text(segments).encode("utf-8"),f"{stem}_raw_timestamped.txt","text/plain",use_container_width=True)
        st.download_button("Download raw subtitles (.srt)",transcript_to_srt(segments).encode("utf-8"),f"{stem}_raw.srt","application/x-subrip",use_container_width=True)
        st.download_button("Download research table (.csv)",segments_to_csv(segments).encode("utf-8-sig"),f"{stem}_segments.csv","text/csv",use_container_width=True)
        report={"source_file":source_name,"health":health,"model":result.get("model"),"verifier_model":result.get("verifier_model"),"validated_chunks":validated}
        st.download_button("Download validation report (.json)",json.dumps(report,ensure_ascii=False,indent=2).encode("utf-8"),f"{stem}_validation.json","application/json",use_container_width=True)
