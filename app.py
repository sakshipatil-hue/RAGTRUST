"""
RAG Reliability Engineer - portfolio dashboard
Reads the seven exported evaluation files in ./data.

Run:   streamlit run app.py
Theme: keep .streamlit/config.toml next to this file (light theme, minimal toolbar).
"""
import html
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------ Edit these ------------------------------
AUTHOR_NAME = "Sakshi"
AUTHOR_TAGLINE = "AI & ML student, IIST Indore"
GITHUB_URL = ""      # e.g. "https://github.com/your-handle/rag-reliability"
LINKEDIN_URL = ""    # e.g. "https://www.linkedin.com/in/your-handle"
PROJECT_TITLE = "RAGTrust — Intelligent RAG Evaluation & Reliability Framework"
PROJECT_TAGLINE = ("Can a small open-source RAG system tell when it does not know? "
                   "An evaluation of grounding, claim checking and abstention on questions about four research papers.")

# Development-set retrieval numbers (not part of the seven exports; edit if you re-run retrieval).
DEV_RETRIEVAL = {
    "Hit@3": {"baseline": 0.375, "improved": 0.375},
    "MRR": {"baseline": 0.3021, "improved": 0.3333},
}
DEV_EVIDENCE_MAPPED, DEV_TOTAL = 16, 20

# Plain-language next steps shown on the Findings page. Edit to match your real plans.
NEXT_STEPS = [
    "Verify claims against several evidence sentences at once, since sentence-by-sentence matching can miss multi-sentence support.",
    "Calibrate the acceptance threshold on the development set so the router is less conservative without accepting unsupported answers.",
    "Add human labels for the correct decision on each question, so routing can be reported as accuracy and not only as counts.",
    "Measure retrieval on the held-out test set as well as on the development set.",
]

st.set_page_config(page_title=f"{PROJECT_TITLE} | {AUTHOR_NAME}", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FILES = {
    "dev_faithfulness": DATA_DIR / "faithfulness_dev_results_v2.json",
    "dev_decisions_csv": DATA_DIR / "decision_routing_dev.csv",
    "dev_decisions_json": DATA_DIR / "decision_routing_dev.json",
    "test_answers": DATA_DIR / "test_answer_results.json",
    "test_faithfulness": DATA_DIR / "faithfulness_test_results.json",
    "test_decisions_csv": DATA_DIR / "decision_routing_test.csv",
    "test_decisions_json": DATA_DIR / "decision_routing_test.json",
}

# ------------------------------ Visual system ------------------------------
INK, MUTED, LINE = "#0B1220", "#64748B", "#E3E7F0"
PRIMARY, VIOLET, CYAN = "#4F46E5", "#7C3AED", "#06B6D4"
GREEN, AMBER, RED, SLATE = "#10B981", "#F59E0B", "#F43F5E", "#94A3B8"
TXT = {GREEN: "#047857", AMBER: "#B45309", RED: "#BE123C", PRIMARY: "#4338CA", SLATE: "#475569", CYAN: "#0E7490", VIOLET: "#6D28D9"}
DECISION_COLORS = {"ACCEPT": GREEN, "HUMAN_REVIEW": AMBER, "ABSTAIN": RED, "UNKNOWN": SLATE}
DECISION_LABELS = {"ACCEPT": "Accepted", "HUMAN_REVIEW": "Sent to human review", "ABSTAIN": "Abstained", "UNKNOWN": "Unknown"}
DECISION_ORDER = ["ACCEPT", "HUMAN_REVIEW", "ABSTAIN"]
CAT_PALETTE = [PRIMARY, CYAN, VIOLET, AMBER, GREEN, RED, SLATE]
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
PAGES = ["Summary", "How it works", "Results", "Explore a question", "Findings"]

ICONS = {
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "flag": '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/>',
    "ban": '<circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "branch": '<line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/>',
}


def icon(name, size=18):
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


def hex_rgba(color, alpha):
    h = color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
html, body, .stApp, [class*="css"] {font-family:'Inter', -apple-system, 'Segoe UI', sans-serif; color:#0B1220;}
.stApp {background:radial-gradient(900px 420px at 85% -80px, rgba(124,58,237,.10), transparent 70%), #F4F6FB;}
#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none!important;}
header[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:1280px; padding-top:1.2rem; padding-bottom:4rem;}
h1, h2, h3, h4, .jk {font-family:'Plus Jakarta Sans','Inter',system-ui,-apple-system,'Segoe UI',sans-serif; letter-spacing:-.02em; color:#0B1220;}

/* sidebar */
section[data-testid="stSidebar"] {background:#FFFFFF; border-right:1px solid #E3E7F0;}
section[data-testid="stSidebar"] .stButton button {width:100%; justify-content:flex-start; border:1px solid #E3E7F0; background:#fff; border-radius:10px; font-weight:600; font-size:13.5px; color:#1E293B;}
section[data-testid="stSidebar"] .stButton button:hover {border-color:#4F46E5; color:#4F46E5; background:#F5F5FF;}
.me {display:flex; gap:12px; align-items:center; margin-bottom:12px;}
.avatar {width:46px; height:46px; border-radius:14px; background:linear-gradient(135deg,#4F46E5,#7C3AED); color:#fff; font-family:'Plus Jakarta Sans','Inter',system-ui,-apple-system,'Segoe UI',sans-serif; font-weight:800; font-size:20px; display:flex; align-items:center; justify-content:center; box-shadow:0 6px 16px rgba(79,70,229,.35);}
.me b {font-family:'Plus Jakarta Sans','Inter',system-ui,-apple-system,'Segoe UI',sans-serif; font-size:16px; display:block;}
.me span {font-size:12.5px; color:#64748B;}
.linkrow a {display:inline-block; font-size:12.5px; font-weight:600; color:#4338CA!important; background:#EEF2FF; border-radius:8px; padding:5px 11px; margin:2px 6px 2px 0; text-decoration:none;}
.side-h {font-family:'Plus Jakarta Sans','Inter',system-ui,-apple-system,'Segoe UI',sans-serif; font-weight:700; font-size:13px; color:#0B1220; margin:14px 0 8px;}

/* hero */
.hero {position:relative; overflow:hidden; border-radius:20px; padding:32px 36px; color:#F8FAFC; background:radial-gradient(circle at 92% 0%, rgba(124,58,237,.65), transparent 45%), radial-gradient(circle at 4% 110%, rgba(6,182,212,.32), transparent 42%), linear-gradient(135deg,#0B1220 0%,#1B1850 100%); display:flex; justify-content:space-between; gap:28px; flex-wrap:wrap; box-shadow:0 16px 40px rgba(15,23,42,.22);}
.hero::after {content:""; position:absolute; inset:0; background-image:linear-gradient(rgba(255,255,255,.045) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.045) 1px, transparent 1px); background-size:36px 36px; mask-image:linear-gradient(90deg, transparent 30%, #000); -webkit-mask-image:linear-gradient(90deg, transparent 30%, #000); pointer-events:none;}
.hero-l, .hero-r {position:relative; z-index:1;}
.hero-l {flex:1 1 540px; min-width:300px;}
.hero h1 {color:#fff; font-size:clamp(30px,3.6vw,44px); font-weight:800; margin:0 0 10px; line-height:1.08; padding:0;}
.hero p {color:#C7D2E8; font-size:15.5px; line-height:1.65; margin:0 0 18px; max-width:660px;}
.chip {display:inline-block; font-size:12px; font-weight:500; color:#E2E8F0; background:rgba(255,255,255,.09); border:1px solid rgba(255,255,255,.18); border-radius:999px; padding:4px 12px; margin:0 6px 6px 0;}
.hero-r {display:flex; flex-direction:column; gap:10px; align-self:center; min-width:230px;}
.hstat {background:rgba(255,255,255,.08); backdrop-filter:blur(6px); border:1px solid rgba(255,255,255,.16); border-radius:14px; padding:12px 18px; display:flex; align-items:baseline; gap:12px;}
.hstat b {font-family:'Plus Jakarta Sans','Inter',system-ui,-apple-system,'Segoe UI',sans-serif; font-size:28px; font-weight:800; line-height:1.1; font-variant-numeric:tabular-nums; min-width:96px;}
.hstat span {font-size:12.5px; color:#C7D2E8; line-height:1.3;}

/* cards */
[class*="st-key-card_"] {background:#FFFFFF; border:1px solid #E3E7F0; border-radius:16px; padding:18px 20px 14px; box-shadow:0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.05);}
.h {font-family:'Plus Jakarta Sans','Inter',system-ui,-apple-system,'Segoe UI',sans-serif; font-size:20px; font-weight:700; margin:28px 0 2px; letter-spacing:-.02em; display:flex; align-items:center;}
.h::before {content:""; width:5px; height:20px; border-radius:3px; background:linear-gradient(#4F46E5,#7C3AED); margin-right:11px;}
.hs {font-size:13.5px; color:#64748B; margin:2px 0 14px 16px; line-height:1.55; max-width:820px;}
.ctitle {font-family:'Plus Jakarta Sans','Inter',system-ui,-apple-system,'Segoe UI',sans-serif; font-size:15.5px; font-weight:700; margin:0 0 2px;}
.csub {font-size:12.5px; color:#64748B; margin:0 0 8px; line-height:1.5;}
.cardtext {font-size:13.5px; color:#334155; line-height:1.6;}
.callout {display:flex; gap:12px; background:linear-gradient(135deg,#EEF2FF,#F5F3FF); border:1px solid #D6DBFF; border-radius:14px; padding:13px 16px; font-size:13.5px; line-height:1.6; color:#312E81; margin:10px 0 14px;}
.callout svg {flex:0 0 18px; margin-top:2px; color:#4F46E5;}

/* grids: step cards and KPI cards */
.grid {display:grid; gap:14px; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); margin:6px 0 4px;}
.scard {background:#fff; border:1px solid #E3E7F0; border-radius:16px; padding:18px 18px 16px; box-shadow:0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.05); transition:transform .18s ease, box-shadow .18s ease;}
.scard:hover {transform:translateY(-3px); box-shadow:0 12px 30px rgba(79,70,229,.14);}
.chipico {width:36px; height:36px; border-radius:11px; display:flex; align-items:center; justify-content:center; background:var(--bg); color:var(--c); margin-bottom:12px;}
.scard b {font-family:'Plus Jakarta Sans','Inter',system-ui,-apple-system,'Segoe UI',sans-serif; font-size:15.5px; display:block; margin-bottom:5px;}
.scard span {font-size:13.5px; color:#475569; line-height:1.6; display:block;}
.grid.k6 {grid-template-columns:repeat(6, 1fr);}
.grid.k4 {grid-template-columns:repeat(4, 1fr);}
.grid.k3 {grid-template-columns:repeat(3, 1fr);}
.grid.k6 .k-val {font-size:27px;}
@media (max-width:1150px) {.grid.k6 {grid-template-columns:repeat(3, 1fr);} .grid.k4 {grid-template-columns:repeat(2, 1fr);}}
@media (max-width:700px) {.grid.k6, .grid.k4, .grid.k3 {grid-template-columns:1fr;}}
.kcard {background:#fff; border:1px solid #E3E7F0; border-radius:16px; padding:16px 18px 15px; box-shadow:0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.05); position:relative; overflow:hidden;}
.kcard::before {content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--c);}
.k-top {display:flex; align-items:center; gap:10px; margin-bottom:10px;}
.k-ico {width:30px; height:30px; border-radius:9px; background:var(--bg); color:var(--c); display:flex; align-items:center; justify-content:center;}
.k-label {font-size:13px; font-weight:600; color:#475569; flex:1;}
.k-help {width:17px; height:17px; border-radius:50%; border:1px solid #CBD5E1; color:#64748B; font-size:11px; font-weight:600; display:flex; align-items:center; justify-content:center; cursor:help;}
.k-val.sm {font-size:21px; line-height:1.3; padding:5px 0 4px;}
.k-val {font-family:'Plus Jakarta Sans','Inter',system-ui,-apple-system,'Segoe UI',sans-serif; font-size:32px; font-weight:800; letter-spacing:-.03em; line-height:1.05; font-variant-numeric:tabular-nums;}
.k-bar {height:6px; border-radius:6px; background:#EEF1F8; margin:11px 0 8px; overflow:hidden;}
.k-bar i {display:block; height:100%; border-radius:6px; background:var(--c); transform-origin:left; animation:grow .9s cubic-bezier(.2,.8,.2,1) both;}
.k-sub {font-size:12.5px; color:#64748B; line-height:1.4;}
@keyframes grow {from {transform:scaleX(0);} to {transform:scaleX(1);}}

/* findings rows */
.frow {display:flex; gap:12px; font-size:14px; line-height:1.6; color:#1E293B; padding:4px 0;}
.dot {flex:0 0 9px; height:9px; border-radius:50%; margin-top:8px; background:linear-gradient(135deg,#4F46E5,#7C3AED);}

/* navigation + controls (pill style) */
[data-testid="stButtonGroup"] [role="radiogroup"] {gap:8px!important; flex-wrap:wrap;}
button[data-variant="segmented_control"] {border-radius:999px!important; border:1px solid #DDE2EE!important; background:#FFFFFF!important; margin:0!important; padding:7px 17px!important; box-shadow:0 1px 2px rgba(16,24,40,.05); transition:all .15s ease;}
button[data-variant="segmented_control"] p {color:#334155!important; font-weight:600; font-size:14px;}
button[data-variant="segmented_control"]:hover {border-color:#4F46E5!important; transform:translateY(-1px);}
button[data-variant="segmented_control"][aria-checked="true"] {background:linear-gradient(135deg,#4F46E5,#7C3AED)!important; border-color:transparent!important; box-shadow:0 8px 18px rgba(79,70,229,.32);}
button[data-variant="segmented_control"][aria-checked="true"] p {color:#FFFFFF!important;}
.st-key-nav {margin:20px 0 4px;}
.st-key-nav button[data-variant="segmented_control"] {padding:10px 22px!important;}
.st-key-nav button[data-variant="segmented_control"] p {font-family:'Plus Jakarta Sans','Inter',system-ui,-apple-system,'Segoe UI',sans-serif; font-weight:700; font-size:15px;}

/* slim header for inner pages */
.slim {display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; border-radius:16px; padding:14px 22px; color:#F8FAFC; background:radial-gradient(circle at 95% 0%, rgba(124,58,237,.55), transparent 50%), linear-gradient(135deg,#0B1220,#1B1850); box-shadow:0 10px 26px rgba(15,23,42,.16);}
.slim b {font-family:'Plus Jakarta Sans','Inter',system-ui,-apple-system,'Segoe UI',sans-serif; font-size:19px; font-weight:800; letter-spacing:-.02em;}
.slim .pills span {display:inline-block; font-size:12.5px; background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.18); border-radius:999px; padding:3px 11px; margin-left:6px; color:#E2E8F0;}

/* evidence tape */
.tape {display:flex; align-items:flex-end; gap:3px; height:120px; border-bottom:1.5px solid #CBD5E1; padding-top:6px;}
.tape i {flex:1; min-width:3px; background:var(--c); height:var(--h); border-radius:4px 4px 0 0; transform-origin:bottom; animation:rise .7s cubic-bezier(.2,.8,.2,1) both; animation-delay:var(--d); cursor:help; transition:filter .15s;}
.tape i:hover {filter:brightness(.82);}
@keyframes rise {from {transform:scaleY(0);} to {transform:scaleY(1);}}
@media (prefers-reduced-motion: reduce) {.tape i, .k-bar i {animation:none;} .scard {transition:none;}}
.key {display:flex; gap:16px; flex-wrap:wrap; font-size:12.5px; color:#475569; margin-top:10px;}
.key u {text-decoration:none; display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px;}

/* pipeline */
.flow {display:grid; gap:12px; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr));}
.fstep {position:relative; background:#fff; border:1px solid #E3E7F0; border-radius:16px; padding:16px 16px 14px; box-shadow:0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.05);}
.fstep::after {content:"\\203A"; position:absolute; right:-10px; top:50%; transform:translateY(-50%); width:20px; height:20px; border-radius:50%; background:#fff; border:1px solid #E3E7F0; color:#64748B; font-size:15px; line-height:17px; text-align:center; z-index:2;}
.fstep:last-child::after {display:none;}
.fstep b {font-family:'Plus Jakarta Sans','Inter',system-ui,-apple-system,'Segoe UI',sans-serif; font-size:14.5px; display:block; margin-bottom:3px;}
.fstep span {font-size:12.5px; color:#64748B; line-height:1.5;}

/* answer x-ray */
.xray {background:#FFFFFF; border:1px solid #E3E7F0; border-radius:14px; padding:18px 20px; font-size:14.5px; line-height:1.9; white-space:pre-wrap; color:#1E293B;}
.xray mark {padding:2px 4px; border-radius:5px; color:#0B1220;}
.xray mark.sup {background:#D1FAE5;}
.xray mark.neu {background:#FEF3C7;}
.xray mark.con {background:#FFE4E6;}
.strip {display:flex; gap:4px; margin:8px 0 4px;}
.strip i {flex:1; height:10px; border-radius:5px; background:var(--c);}
.claimrow {border:1px solid #E3E7F0; border-left:4px solid var(--c); background:#fff; border-radius:12px; padding:10px 14px; margin:10px 0;}
.claimrow b {font-size:12.5px; color:var(--t);}
.claimrow div {font-size:13.5px; line-height:1.55; margin-top:2px; color:#1E293B;}
.claimrow small {color:#64748B; font-size:12px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------ Data helpers ------------------------------
def esc(x):
    return html.escape(str(x if x is not None else ""))


def read_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_records(obj):
    """Accept a JSON list or common dict wrappers containing a list."""
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("results", "data", "records", "items", "questions"):
            if isinstance(obj.get(key), list):
                return obj[key]
        if any(k in obj for k in ("Question_ID", "question_id", "qid")):
            return [obj]
        rows = []
        for key, value in obj.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("Question_ID", key)
                rows.append(row)
        return rows
    return []


def first(row, *keys, default=None):
    for key in keys:
        if isinstance(row, dict) and key in row and row[key] is not None:
            return row[key]
    return default


def safe_float(value):
    try:
        v = float(value)
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return None


def safe_list(value):
    return value if isinstance(value, list) else []


def load_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def normalize_decision(value):
    value = str(value or "UNKNOWN").strip().upper().replace(" ", "_")
    return "HUMAN_REVIEW" if value in ("HUMANREVIEW", "REVIEW") else value


def by_qid(rows):
    out = {}
    for row in rows:
        qid = str(first(row, "Question_ID", "question_id", "qid", "ID", default=""))
        if qid:
            out[qid] = row
    return out


def normalize_decision_df(csv_df, json_rows):
    rows = [r.to_dict() for _, r in csv_df.iterrows()] if not csv_df.empty else json_rows
    out = []
    for row in rows:
        qid = str(first(row, "Question_ID", "question_id", "qid", "ID", default=""))
        if not qid:
            continue
        out.append({
            "Question_ID": qid,
            "Decision": normalize_decision(first(
                row, "Decision", "decision", "Final_Decision", "Action", "action", "Routing_Decision", default="UNKNOWN")),
            "Answerability": str(first(row, "Answerability", "answerability", default="Unknown")),
            "Total_Claims": safe_float(first(row, "Total_Claims", "total_claims")),
            "Supported_Claims": safe_float(first(row, "Supported_Claims", "supported_claims")),
            "Faithfulness_Score": safe_float(first(row, "Faithfulness_Score", "faithfulness_score")),
        })
    return pd.DataFrame(out)


def merge_split(answer_rows, faith_rows, decision_df):
    answers, faith = by_qid(answer_rows), by_qid(faith_rows)
    decision = {str(r["Question_ID"]): r.to_dict() for _, r in decision_df.iterrows()} if not decision_df.empty else {}
    merged = []
    for qid in sorted(set(answers) | set(faith) | set(decision)):
        a, f, d = answers.get(qid, {}), faith.get(qid, {}), decision.get(qid, {})
        total = first(f, "Total_Claims", "total_claims", default=d.get("Total_Claims"))
        supported = first(f, "Supported_Claims", "supported_claims", default=d.get("Supported_Claims"))
        score = first(f, "Faithfulness_Score", "faithfulness_score", default=d.get("Faithfulness_Score"))
        merged.append({
            "Question_ID": qid,
            "Question": first(a, "Question", "question", default=first(d, "Question", "question", default="")),
            "Category": first(a, "Category", "category", default="Unknown"),
            "Answerability": first(a, "Answerability", "answerability", default=d.get("Answerability", "Unknown")),
            "Reference_Answer": first(a, "Reference_Answer", "reference_answer", default=""),
            "Generated_Answer": first(a, "Generated_Answer", "generated_answer", "Answer", "answer", default=""),
            "Latency_Seconds": safe_float(first(a, "Latency_Seconds", "latency_seconds")),
            "Status": first(a, "Status", "status", default=""),
            "Total_Claims": safe_float(total),
            "Supported_Claims": safe_float(supported),
            "Faithfulness_Score": safe_float(score),
            "Decision": normalize_decision(first(d, "Decision", "decision", default="UNKNOWN")),
            "Claim_Details": safe_list(first(f, "Claim_Details", "claim_details", "Claims", "claims", default=[])),
            "Retrieved_Chunks": safe_list(first(a, "Retrieved_Chunks", "retrieved_chunks", "chunks", default=[])),
        })
    return pd.DataFrame(merged)


def verdict_code(claim):
    """2 = supported, 1 = neutral / needs review, 0 = contradicted or unsupported."""
    v = str(first(claim, "verdict", "Verdict", default="")).upper() if isinstance(claim, dict) else ""
    if v == "SUPPORTED":
        return 2
    if v in ("NEUTRAL", "NEEDS_REVIEW"):
        return 1
    return 0


def grounding_band(score, total):
    if total is None or pd.isna(total) or total == 0 or score is None or pd.isna(score):
        return "No claims"
    if score < 1 / 3:
        return "Low grounding"
    if score < 2 / 3:
        return "Partial grounding"
    return "Strong grounding"


def pct(value, digits=1):
    return "n/a" if value is None or pd.isna(value) else f"{value * 100:.{digits}f}%"


def is_answerable(s):
    return str(s).strip().lower() == "answerable"


def is_unanswerable(s):
    return str(s).strip().lower() == "unanswerable"


def hex_rgba(color, alpha):
    h = color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


# ------------------------------ UI helpers ------------------------------
def keyed(name):
    """Container with a CSS-addressable key. Falls back to a plain container on older Streamlit versions."""
    try:
        return st.container(key=name)
    except TypeError:
        return st.container()


def card(name):
    return keyed(f"card_{name}")


def show(fig, height=340, legend=True):
    fig.update_layout(
        height=height, margin=fig.layout.margin if fig.layout.margin.l is not None else dict(l=10, r=14, t=16, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=INK, size=12),
        showlegend=legend, legend=dict(orientation="h", y=-0.2, title_text=""),
        hoverlabel=dict(font_family="Inter, sans-serif", bgcolor="#0B1220", font_color="#fff"),
        bargap=0.38,
    )
    fig.update_xaxes(gridcolor="#EEF1F8", zerolinecolor=LINE, linecolor=LINE, automargin=True)
    fig.update_yaxes(gridcolor="#EEF1F8", zerolinecolor=LINE, linecolor=LINE, automargin=True)
    try:
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, theme=None)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, theme=None)


def table(df, **kwargs):
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kwargs)
    except Exception:
        st.dataframe(df, hide_index=True, use_container_width=True, **kwargs)


def heading(title, sub=""):
    sub_html = f'<div class="hs">{sub}</div>' if sub else ""
    st.markdown(f'<div class="h">{title}</div>{sub_html}', unsafe_allow_html=True)


def ctitle(title, sub=""):
    st.markdown(f'<div class="ctitle">{title}</div><div class="csub">{sub}</div>', unsafe_allow_html=True)


def callout(text):
    st.markdown(f'<div class="callout">{icon("flag", 18)}<div>{text}</div></div>', unsafe_allow_html=True)


def info_cards(items):
    """items: (icon, color, title, text). Equal-height grid cards."""
    cells = "".join(
        f'<div class="scard"><div class="chipico" style="--c:{c};--bg:{hex_rgba(c, .12)}">{icon(i, 19)}</div><b>{t}</b><span>{d}</span></div>'
        for i, c, t, d in items)
    st.markdown(f'<div class="grid">{cells}</div>', unsafe_allow_html=True)


def kpi_cards(items):
    """items: (icon, color, label, value, sub, help, fill 0..1 or None)."""
    cells = []
    for i, c, label, value, sub, help_, fill in items:
        bar = f'<div class="k-bar"><i style="width:{max(0, min(1, fill)) * 100:.0f}%"></i></div>' if fill is not None else '<div style="height:10px"></div>'
        cells.append(
            f'<div class="kcard" style="--c:{c};--bg:{hex_rgba(c, .12)}"><div class="k-top"><div class="k-ico">{icon(i, 16)}</div>'
            f'<div class="k-label">{label}</div><div class="k-help" title="{esc(help_)}">?</div></div>'
            f'<div class="k-val{" sm" if len(str(value)) > 11 else ""}">{value}</div>{bar}<div class="k-sub">{sub}</div></div>')
    cls = f"k{len(items)}" if len(items) in (3, 4, 6) else ""
    st.markdown(f'<div class="grid {cls}">{"".join(cells)}</div>', unsafe_allow_html=True)


def seg(label, options, key, fmt=None):
    """Pill selector that survives being clicked twice (which would otherwise deselect it)."""
    last = st.session_state.get("_last_" + key, options[0])
    if st.session_state.get(key) not in options:
        st.session_state[key] = last if last in options else options[0]
    if hasattr(st, "segmented_control"):
        kw = {"key": key, "label_visibility": "collapsed"}
        if fmt:
            kw["format_func"] = fmt
        v = st.segmented_control(label, options, **kw)
    else:
        v = st.radio(label, options, key=key, horizontal=True, label_visibility="collapsed", format_func=fmt or str)
    v = v if v in options else (last if last in options else options[0])
    st.session_state["_last_" + key] = v
    return v


def columns_centered(widths):
    try:
        return st.columns(widths, vertical_alignment="center")
    except TypeError:
        return st.columns(widths)


def go_to(page, view=None):
    st.session_state["page"] = page
    if view:
        st.session_state["rview"] = view


def evidence_tape(df, color_by, sort_by):
    """One bar per question: height = share of claims supported, colour = chosen grouping."""
    d = df.copy()
    if sort_by == "Grounding, high to low":
        d = d.sort_values("Faithfulness_Score", ascending=False, na_position="last")
    if color_by == "Routing decision":
        colors = lambda r: DECISION_COLORS.get(r["Decision"], SLATE)
        legend = [(DECISION_LABELS[k], DECISION_COLORS[k]) for k in DECISION_ORDER]
    else:
        col = "Answerability" if color_by == "Answerability" else "Category"
        values = sorted(d[col].astype(str).unique())
        cmap = {v: CAT_PALETTE[i % len(CAT_PALETTE)] for i, v in enumerate(values)}
        colors = lambda r: cmap[str(r[col])]
        legend = [(v, c) for v, c in cmap.items()]
    bars = []
    for i, (_, r) in enumerate(d.iterrows()):
        score = r["Faithfulness_Score"]
        height = 4 if score is None or pd.isna(score) else max(6, round(score * 100))
        sup = "n/a" if pd.isna(r["Supported_Claims"]) else int(r["Supported_Claims"])
        tot = "n/a" if pd.isna(r["Total_Claims"]) else int(r["Total_Claims"])
        tip = (f'{r["Question_ID"]}: {DECISION_LABELS.get(r["Decision"], r["Decision"])}, {r["Answerability"]}, '
               f'{sup} of {tot} claims supported')
        bars.append(f'<i style="--c:{colors(r)};--h:{height}%;--d:{i * 18}ms" title="{esc(tip)}"></i>')
    key = "".join(f'<span><u style="background:{c}"></u>{esc(n)}</span>' for n, c in legend)
    st.markdown(f'<div class="tape">{"".join(bars)}</div><div class="key">{key}</div>', unsafe_allow_html=True)


def xray(answer, claims):
    text = esc(answer) or "No generated answer found."
    located = 0
    for c in claims:
        if not isinstance(c, dict):
            continue
        ctext = esc(first(c, "claim", "Claim", "text", default=""))
        if ctext and ctext in text:
            cls = {2: "sup", 1: "neu", 0: "con"}[verdict_code(c)]
            text = text.replace(ctext, f'<mark class="{cls}">{ctext}</mark>', 1)
            located += 1
    return text, located


def reliability_profile(df):
    """Four 0..1 readings used by the radar. Unanswerable questions should be abstained by benchmark design."""
    if df is None or df.empty or "Decision" not in df:
        return {}
    out = {}
    tot, sup = df["Total_Claims"].fillna(0).sum(), df["Supported_Claims"].fillna(0).sum()
    if tot:
        out["Claim grounding"] = sup / tot
    out["Answers with some support"] = float((df["Supported_Claims"].fillna(0) > 0).mean())
    una = df[df["Answerability"].map(is_unanswerable)]
    ans = df[df["Answerability"].map(is_answerable)]
    if len(una):
        out["Withholds unanswerable"] = float((una["Decision"] == "ABSTAIN").mean())
    if len(ans):
        out["Answers answerable"] = float(1 - (ans["Decision"] == "ABSTAIN").mean())
    return out


# ------------------------------ Load data ------------------------------
missing = [name for name, path in FILES.items() if not path.exists()]
if missing:
    st.error("Some dashboard data files are missing from the data folder.")
    for name in missing:
        st.write(f"- `{FILES[name].name}`")
    st.info("Copy all seven exported files into the data folder and keep their filenames unchanged.")
    st.stop()

try:
    dev_decisions = normalize_decision_df(load_csv(FILES["dev_decisions_csv"]), as_records(read_json(FILES["dev_decisions_json"])))
    test_decisions = normalize_decision_df(load_csv(FILES["test_decisions_csv"]), as_records(read_json(FILES["test_decisions_json"])))
    test_answer_rows = as_records(read_json(FILES["test_answers"]))
    test_faith_rows = as_records(read_json(FILES["test_faithfulness"]))
    dev_df = merge_split([], as_records(read_json(FILES["dev_faithfulness"])), dev_decisions)
    test_df = merge_split(test_answer_rows, test_faith_rows, test_decisions)
except Exception as exc:
    st.error(f"Could not read the exported evaluation data: {exc}")
    st.exception(exc)
    st.stop()

if test_df.empty:
    st.error("The test files loaded, but they contain no question records.")
    st.stop()

if not dev_df.empty and not dev_decisions.empty:
    dev_df = dev_df.drop(columns=["Decision", "Answerability"], errors="ignore").merge(
        dev_decisions[["Question_ID", "Decision", "Answerability"]], on="Question_ID", how="left")
test_df["Band"] = [grounding_band(s, t) for s, t in zip(test_df["Faithfulness_Score"], test_df["Total_Claims"])]

# ------------------------------ Headline numbers (full test set) ------------------------------
N = len(test_df)
all_claims = int(test_df["Total_Claims"].fillna(0).sum())
all_sup = int(test_df["Supported_Claims"].fillna(0).sum())
all_faith = all_sup / all_claims if all_claims else None
dev_claims = int(dev_df["Total_Claims"].fillna(0).sum()) if not dev_df.empty else 0
dev_sup = int(dev_df["Supported_Claims"].fillna(0).sum()) if not dev_df.empty else 0
dev_faith = dev_sup / dev_claims if dev_claims else None
dc = test_df["Decision"].value_counts()
A_ACC, A_REV, A_ABS = (int(dc.get(k, 0)) for k in DECISION_ORDER)
ok_count = int((test_df["Status"].astype(str).str.lower() == "success").sum()) or N
avg_latency = test_df["Latency_Seconds"].dropna().mean()
all_answerable = test_df[test_df["Answerability"].map(is_answerable)]
all_unanswerable = test_df[test_df["Answerability"].map(is_unanswerable)]
answerable_abstained = int((all_answerable["Decision"] == "ABSTAIN").sum())
unanswerable_withheld = int((all_unanswerable["Decision"] == "ABSTAIN").sum())

# ------------------------------ Sidebar ------------------------------
with st.sidebar:
    initial = esc(AUTHOR_NAME[:1].upper() or "?")
    links = ""
    if GITHUB_URL:
        links += f'<a href="{esc(GITHUB_URL)}" target="_blank">GitHub</a>'
    if LINKEDIN_URL:
        links += f'<a href="{esc(LINKEDIN_URL)}" target="_blank">LinkedIn</a>'
    st.markdown(f'<div class="me"><div class="avatar">{initial}</div><div><b>{esc(AUTHOR_NAME)}</b>'
                f'<span>{esc(AUTHOR_TAGLINE)}</span></div></div><div class="linkrow">{links}</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-h">Jump to</div>', unsafe_allow_html=True)
    for p_, label in [("Summary", "Summary, the 60-second version"), ("How it works", "How it works"),
                      ("Results", "Results and charts"), ("Explore a question", "Explore one answer"), ("Findings", "Findings and next steps")]:
        st.button(label, key=f"side_{p_}", on_click=go_to, args=(p_,))
    st.markdown('<div class="side-h">About the data</div>', unsafe_allow_html=True)
    st.caption(f"{len(dev_df)} development and {N} held-out test questions. "
               f"{len(test_answer_rows)} answer records and {len(test_faith_rows)} claim records loaded.")

# ------------------------------ Header ------------------------------
current_page = st.session_state.get("page") or "Summary"
chips = "".join(f'<span class="chip">{c}</span>' for c in
                ["Qwen2.5-3B-Instruct", "FAISS", "MiniLM embeddings", "Cross-encoder reranker", "DeBERTa NLI verifier", "Streamlit"])
if current_page == "Summary":
    st.markdown(
        f'<div class="hero"><div class="hero-l"><h1>{esc(PROJECT_TITLE)}</h1><p>{esc(PROJECT_TAGLINE)}</p>{chips}</div>'
        f'<div class="hero-r">'
        f'<div class="hstat"><b>{pct(all_faith)}</b><span>of claims supported by evidence</span></div>'
        f'<div class="hstat"><b>{N}</b><span>held-out test questions</span></div>'
        f'<div class="hstat"><b>{A_ACC} / {A_REV} / {A_ABS}</b><span>accepted, review, abstained</span></div>'
        f'</div></div>', unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="slim"><b>{esc(PROJECT_TITLE)}</b><div class="pills"><span>{pct(all_faith)} of claims supported</span>'
        f'<span>{N} test questions</span><span>{A_ACC} / {A_REV} / {A_ABS} accepted, review, abstained</span></div></div>',
        unsafe_allow_html=True)

with keyed("nav"):
    page = seg("Page", PAGES, "page")

# =================================== SUMMARY ===================================
if page == "Summary":
    heading("The project in 60 seconds")
    info_cards([
        ("target", PRIMARY, "The problem", "Small language models can sound sure while being wrong. This project measures how much of an answer is actually backed by its sources."),
        ("layers", VIOLET, "The system", "It retrieves passages from four research papers, writes an answer, splits it into claims, and checks each claim with an NLI model."),
        ("shield", GREEN, "The decision", "Each answer is accepted, sent to a human reviewer, or withheld. The router is deliberately cautious."),
        ("database", CYAN, "The benchmark", f"60 questions of five types. {len(dev_df)} were used to tune the system and {N} were held out for the final test."),
    ])

    heading("Results at a glance", "Held-out test set. Hover the ? on a card for what it means.")
    delta_txt = ""
    if dev_faith is not None and all_faith is not None:
        delta_txt = f"{(all_faith - dev_faith) * 100:+.1f} pts vs development ({pct(dev_faith)})"
    kpi_cards([
        ("check", GREEN, "Claim grounding", pct(all_faith), delta_txt or f"{all_sup} of {all_claims} claims",
         "Share of extracted claims that the NLI verifier found supported by the retrieved evidence.", all_faith or 0),
        ("file", PRIMARY, "Answers generated", f"{ok_count}/{N}", "Questions answered without error",
         "Questions for which the model produced an answer without error.", ok_count / N),
        ("shield", GREEN, "Accepted", str(A_ACC), f"{A_ACC / N * 100:.0f}% of questions",
         "Answers released automatically because they passed the acceptance rule.", A_ACC / N),
        ("flag", AMBER, "Sent to review", str(A_REV), f"{A_REV / N * 100:.0f}% of questions",
         "Answers flagged for a human to read before use.", A_REV / N),
        ("ban", RED, "Abstained", str(A_ABS), f"{A_ABS / N * 100:.0f}% of questions",
         "Questions where the system declined to answer.", A_ABS / N),
        ("clock", CYAN, "Average latency", f"{avg_latency:.1f}s" if pd.notna(avg_latency) else "n/a", "Per answer, on a Colab T4",
         "Mean time to generate one answer.", None),
    ])

    heading("Every test question at a glance", "Each bar is one question. Taller means more of its claims were backed by evidence. Change the colouring or order, then hover a bar.")
    with card("tape"):
        t1, t2, _ = st.columns([2.3, 2.1, 0.4])
        with t1:
            st.caption("Colour by")
            color_by = seg("Colour by", ["Routing decision", "Answerability", "Question type"], "tape_color")
        with t2:
            st.caption("Order")
            sort_by = seg("Order", ["Question ID", "Grounding, high to low"], "tape_sort")
        evidence_tape(test_df, color_by, sort_by)

    c1, c2 = st.columns([1, 1.25], gap="medium")
    with c1:
        with card("donut"):
            ctitle("How questions were routed", "Final decision for each held-out question.")
            fig = go.Figure(go.Pie(labels=[DECISION_LABELS[k_] for k_ in DECISION_ORDER], values=[A_ACC, A_REV, A_ABS], hole=0.68, sort=False,
                                   marker=dict(colors=[GREEN, AMBER, RED], line=dict(color="#fff", width=3)), textinfo="value"))
            fig.add_annotation(text=f"<b>{N}</b><br>questions", showarrow=False, font=dict(size=16, color=INK))
            show(fig, 290)
    with c2:
        with card("radar"):
            ctitle("Reliability profile", "Higher is better on every axis. Unanswerable questions should be withheld and answerable ones should not, by benchmark design.")
            pt, pd_ = reliability_profile(test_df), reliability_profile(dev_df)
            axes = [a for a in pt if a in pd_] if pd_ else list(pt)
            short = {"Claim grounding": "Grounding", "Answers with some support": "Some support", "Withholds unanswerable": "Withholds unanswerable", "Answers answerable": "Answers answerable"}
            if len(axes) >= 3:
                fig = go.Figure()
                if pd_:
                    fig.add_trace(go.Scatterpolar(r=[pd_[a] for a in axes] + [pd_[axes[0]]], theta=[short.get(a, a) for a in axes] + [short.get(axes[0], axes[0])], fill="toself", name="Development",
                                                  line=dict(color=CYAN, width=2), fillcolor=hex_rgba(CYAN, .16)))
                fig.add_trace(go.Scatterpolar(r=[pt[a] for a in axes] + [pt[axes[0]]], theta=[short.get(a, a) for a in axes] + [short.get(axes[0], axes[0])], fill="toself", name="Held-out test",
                                              line=dict(color=PRIMARY, width=2.5), fillcolor=hex_rgba(PRIMARY, .22)))
                fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 1], tickvals=[0.5, 1], tickformat=".0%", gridcolor="#E3E7F0", angle=45, tickfont=dict(size=10)),
                                             angularaxis=dict(gridcolor="#E3E7F0")))
                fig.update_layout(margin=dict(l=90, r=90, t=30, b=10))
                show(fig, 290)
            else:
                st.info("Not enough answerability data to draw the profile.")

    heading("What stood out", "Each finding links to the chart behind it.")
    cat_scores = test_df[test_df["Total_Claims"].fillna(0) > 0].groupby("Category")["Faithfulness_Score"].mean().dropna()
    d_mrr = DEV_RETRIEVAL["MRR"]["improved"] - DEV_RETRIEVAL["MRR"]["baseline"]
    findings = []
    if all_claims:
        extra = f" The development set scored {pct(dev_faith)}." if dev_faith is not None else ""
        findings.append((f"<b>{all_sup} of {all_claims} claims</b> ({pct(all_faith)}) were supported by retrieved evidence on the held-out set.{extra}", "Results", "Grounding", "See grounding"))
    findings.append(("No question was accepted automatically, so the acceptance rule is very conservative." if A_ACC == 0
                     else f"{A_ACC} questions were accepted automatically. They deserve a manual check for unsafe accepts.", "Results", "Routing", "See routing"))
    if len(all_unanswerable):
        findings.append((f"<b>{unanswerable_withheld} of {len(all_unanswerable)}</b> unanswerable questions were withheld, which is the intended behaviour. "
                         f"<b>{answerable_abstained} of {len(all_answerable)}</b> answerable questions were also abstained, and some of those may be false abstentions.", "Explore a question", None, "Read them"))
    if len(cat_scores) >= 2 and cat_scores.index.astype(str).tolist() != ["Unknown"]:
        findings.append((f"Grounding was highest for <b>{esc(cat_scores.idxmax())}</b> questions ({pct(cat_scores.max())} average) and lowest for <b>{esc(cat_scores.idxmin())}</b> ({pct(cat_scores.min())}).", "Results", "Grounding", "See grounding"))
    findings.append((f"On the development set, reranking raised MRR from {DEV_RETRIEVAL['MRR']['baseline']:.4f} to {DEV_RETRIEVAL['MRR']['improved']:.4f}, while Hit@3 stayed at {DEV_RETRIEVAL['Hit@3']['improved']:.3f}.", "Results", "Retrieval", "See retrieval"))
    with card("findings"):
        for i, (txt, pg, vw, label) in enumerate(findings):
            a, b = columns_centered([7, 1.3])
            a.markdown(f'<div class="frow"><div class="dot"></div><div>{txt}</div></div>', unsafe_allow_html=True)
            b.button(label, key=f"find_{i}", on_click=go_to, args=(pg, vw))
    callout("This is a research prototype. The numbers describe how the verifier and router behaved on a 40-question benchmark. They are not a guarantee of reliability.")

# =================================== HOW IT WORKS ===================================
elif page == "How it works":
    heading("Pipeline", "Each question passes through six stages. The last two decide whether an answer can be trusted.")
    stages = [("database", SLATE, "Corpus", "4 research PDFs split into 499 chunks"),
              ("search", PRIMARY, "Retrieve", "FAISS search over MiniLM embeddings"),
              ("branch", VIOLET, "Expand and rerank", "Multi-query search, then cross-encoder rerank"),
              ("zap", CYAN, "Generate", "Qwen2.5-3B-Instruct writes the answer"),
              ("check", AMBER, "Verify", "DeBERTa NLI checks every claim against evidence"),
              ("shield", GREEN, "Route", "Accept, send to review, or abstain")]
    st.markdown('<div class="flow">' + "".join(
        f'<div class="fstep"><div class="chipico" style="--c:{c};--bg:{hex_rgba(c, .12)}">{icon(i, 19)}</div><b>{n_}</b><span>{d_}</span></div>'
        for i, c, n_, d_ in stages) + '</div>', unsafe_allow_html=True)

    heading("Benchmark and stack")
    left, right = st.columns([1.2, 1], gap="medium")
    with left:
        with card("compo"):
            ctitle("Held-out test set composition", "Number of questions by type, split by whether the corpus can answer them.")
            comp = test_df.groupby(["Category", "Answerability"]).size().reset_index(name="Questions")
            fig = px.bar(comp, y="Category", x="Questions", color="Answerability", orientation="h", color_discrete_sequence=[PRIMARY, AMBER, GREEN, SLATE])
            fig.update_yaxes(title="", autorange="reversed")
            show(fig, 300)
        with card("types"):
            ctitle("The five question types")
            st.markdown(
                '<div class="cardtext"><b>Single fact</b>: one fact from one passage.<br>'
                '<b>Multi-hop</b>: needs evidence from several passages or papers.<br>'
                '<b>Unanswerable</b>: the corpus does not contain the answer, so the right move is to abstain.<br>'
                '<b>Partially answerable</b>: part of the question can be answered and part cannot.<br>'
                '<b>Cross-paper disagreement</b>: the papers use different definitions or assumptions, and a good answer says so.</div>', unsafe_allow_html=True)
    with right:
        with card("stack"):
            ctitle("Technology stack")
            table(pd.DataFrame([
                ("Language model", "Qwen/Qwen2.5-3B-Instruct"), ("Embeddings", "sentence-transformers/all-MiniLM-L6-v2"),
                ("Vector store", "FAISS"), ("Reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                ("Claim verifier", "cross-encoder/nli-deberta-v3-base"), ("Runtime", "Google Colab, NVIDIA T4"),
                ("Dashboard", "Streamlit and Plotly")], columns=["Component", "Technology"]))

# =================================== RESULTS ===================================
elif page == "Results":
    with st.expander("Filter the results", expanded=False):
        f1, f2, f3 = st.columns(3)
        cats = f1.multiselect("Question type", sorted(test_df["Category"].astype(str).unique()))
        ans = f2.multiselect("Answerability", sorted(test_df["Answerability"].astype(str).unique()))
        dec = f3.multiselect("Routing decision", [d for d in DECISION_ORDER + ["UNKNOWN"] if d in set(test_df["Decision"])],
                             format_func=lambda d: DECISION_LABELS.get(d, d))
    fdf = test_df.copy()
    if cats:
        fdf = fdf[fdf["Category"].astype(str).isin(cats)]
    if ans:
        fdf = fdf[fdf["Answerability"].astype(str).isin(ans)]
    if dec:
        fdf = fdf[fdf["Decision"].isin(dec)]

    if fdf.empty:
        st.warning("No questions match these filters. Clear a filter above.")
    else:
        n = len(fdf)
        claims = int(fdf["Total_Claims"].fillna(0).sum())
        supported = int(fdf["Supported_Claims"].fillna(0).sum())
        faith = supported / claims if claims else None
        st.caption(f"Showing {n} of {N} test questions. {supported} of {claims} claims supported ({pct(faith)}).")
        view = seg("Section", ["Grounding", "Routing", "Retrieval"], "rview")

        if view == "Grounding":
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                with card("devtest"):
                    ctitle("Development against held-out test", "The test set was not used for tuning, so a gap shows how much tuning transferred.")
                    labels, vals, cols_ = [], [], []
                    if dev_faith is not None:
                        labels.append("Development"); vals.append(dev_faith * 100); cols_.append(CYAN)
                    labels.append("Held-out test"); vals.append((faith or 0) * 100); cols_.append(PRIMARY)
                    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=cols_,
                                           text=[f"{v:.1f}%" for v in vals], textposition="outside", cliponaxis=False))
                    fig.update_xaxes(range=[0, 100], title="Claims supported (%)")
                    fig.update_yaxes(autorange="reversed")
                    show(fig, 240, legend=False)
            with c2:
                with card("funnel"):
                    ctitle("Where questions drop out", "Each step keeps only questions that cleared the one before it.")
                    has_claims = int((fdf["Total_Claims"].fillna(0) > 0).sum())
                    has_support = int((fdf["Supported_Claims"].fillna(0) > 0).sum())
                    n_acc = int((fdf["Decision"] == "ACCEPT").sum())
                    n_ok = int((fdf["Status"].astype(str).str.lower() == "success").sum()) or n
                    stp = [("Asked", n), ("Answered", min(n, n_ok)), ("Has claims", has_claims), ("Has a supported claim", has_support), ("Accepted", n_acc)]
                    fig = go.Figure(go.Funnel(y=[s[0] for s in stp], x=[s[1] for s in stp], textinfo="value+percent initial",
                                              marker=dict(color=[PRIMARY, "#6366F1", "#818CF8", AMBER, GREEN]),
                                              connector=dict(fillcolor="#EEF1F8", line=dict(width=0))))
                    show(fig, 240, legend=False)

            with card("matrix"):
                ctitle("Claim matrix", "One row per question, one square per claim. Green was supported, yellow neutral, red contradicted or unmatched. Hover a square to read the claim.")
                mrows = fdf[fdf["Total_Claims"].fillna(0) > 0]
                if mrows.empty:
                    st.info("No claim-level records in this view.")
                else:
                    grid, hover, ids, from_counts = [], [], [], 0
                    for _, r in mrows.iterrows():
                        det = [c for c in r["Claim_Details"] if isinstance(c, dict)]
                        if det:
                            codes = [verdict_code(c) for c in det]
                            txt = [esc(str(first(c, "claim", "Claim", "text", default=""))[:140]) for c in det]
                        else:
                            tot, sup = int(r["Total_Claims"]), int(r["Supported_Claims"] or 0)
                            codes = [2] * sup + [0] * (tot - sup)
                            txt = ["(claim text not exported)"] * tot
                            from_counts += 1
                        grid.append(codes); hover.append(txt); ids.append(r["Question_ID"])
                    width = max(len(g_) for g_ in grid)
                    z = [g_ + [None] * (width - len(g_)) for g_ in grid]
                    t = [h_ + [""] * (width - len(h_)) for h_ in hover]
                    cm = go.Figure(go.Heatmap(z=z, x=[f"Claim {i + 1}" for i in range(width)], y=ids, text=t, xgap=3, ygap=3,
                                              showscale=False, colorscale=[[0, RED], [0.5, AMBER], [1, GREEN]], zmin=0, zmax=2,
                                              hovertemplate="%{y}, %{x}<br>%{text}<extra></extra>"))
                    cm.update_yaxes(autorange="reversed", type="category")
                    cm.update_xaxes(side="top")
                    show(cm, max(320, 20 * len(ids) + 80), legend=False)
                    if from_counts:
                        st.caption(f"{from_counts} question(s) had no claim details in the export, so their squares come from counts, supported first. Order there carries no meaning.")

            c3, c4 = st.columns(2, gap="medium")
            with c3:
                with card("scatter"):
                    ctitle("Speed against grounding", "Bubble size is the number of claims. Hover for the question ID.")
                    sc = fdf.dropna(subset=["Latency_Seconds", "Faithfulness_Score"]).copy()
                    if sc.empty:
                        st.info("Latency or grounding values are missing for this view.")
                    else:
                        sc["Grounding_pct"] = sc["Faithfulness_Score"] * 100
                        sc["Claims"] = sc["Total_Claims"].fillna(0).clip(lower=1)
                        fig = px.scatter(sc, x="Latency_Seconds", y="Grounding_pct", size="Claims", size_max=22, color="Decision",
                                         color_discrete_map=DECISION_COLORS, hover_name="Question_ID",
                                         hover_data={"Decision": False, "Claims": False, "Latency_Seconds": ":.2f", "Grounding_pct": ":.1f"})
                        for tr in fig.data:
                            tr.name = DECISION_LABELS.get(tr.name, tr.name)
                            tr.marker.line = dict(color="#fff", width=1.5)
                            tr.marker.opacity = 0.85
                        fig.update_xaxes(title="Latency (seconds)")
                        fig.update_yaxes(title="Claims supported (%)", range=[-5, 105])
                        show(fig, 340)
            with c4:
                with card("box"):
                    ctitle("Grounding by answerability", "Each dot is a question.")
                    bx = fdf.dropna(subset=["Faithfulness_Score"]).copy()
                    if bx.empty:
                        st.info("No grounding scores in this view.")
                    else:
                        bx["Grounding_pct"] = bx["Faithfulness_Score"] * 100
                        fig = px.box(bx, x="Answerability", y="Grounding_pct", color="Answerability", points="all", hover_name="Question_ID",
                                     color_discrete_sequence=[PRIMARY, AMBER, GREEN, SLATE])
                        fig.update_yaxes(title="Claims supported (%)", range=[-5, 105])
                        show(fig, 340, legend=False)

            ranked = fdf[fdf["Total_Claims"].fillna(0) > 0].sort_values("Faithfulness_Score", ascending=False)
            cols_t = ["Question_ID", "Question", "Faithfulness_Score"]
            cfg2 = {"Faithfulness_Score": st.column_config.ProgressColumn("Grounding", min_value=0.0, max_value=1.0, format="%.2f")}
            t1, t2 = st.columns(2, gap="medium")
            with t1:
                with card("best"):
                    ctitle("Best grounded answers")
                    table(ranked.head(8)[cols_t], column_config=cfg2)
            with t2:
                with card("worst"):
                    ctitle("Worst grounded answers")
                    table(ranked.tail(8).iloc[::-1][cols_t], column_config=cfg2)
            st.download_button("Download this view as CSV", fdf.drop(columns=["Claim_Details", "Retrieved_Chunks"], errors="ignore").to_csv(index=False),
                               "rag_reliability_view.csv", "text/csv")

        elif view == "Routing":
            with card("sankey"):
                ctitle("How questions flowed through the router",
                       "Follow a question from what it asked for, to the decision it received, to how well its answer was grounded.")
                node_keys, node_idx = [], {}

                def node(layer, label):
                    key = (layer, label)
                    if key not in node_idx:
                        node_idx[key] = len(node_keys)
                        node_keys.append(key)
                    return node_idx[key]

                src, tgt, val, lcol = [], [], [], []
                band_color = {"Strong grounding": GREEN, "Partial grounding": AMBER, "Low grounding": RED, "No claims": SLATE}
                grp = fdf.assign(A=fdf["Answerability"].astype(str), D=fdf["Decision"], B=fdf["Band"]).groupby(["A", "D", "B"]).size()
                for (l0, l1, l2), cnt in grp.items():
                    a, d, b = node(0, l0), node(1, l1), node(2, l2)
                    for s_, t_, c_ in ((a, d, DECISION_COLORS.get(l1, SLATE)), (d, b, band_color.get(l2, SLATE))):
                        src.append(s_); tgt.append(t_); val.append(cnt); lcol.append(hex_rgba(c_, 0.32))
                node_labels = [DECISION_LABELS.get(l, l) if layer == 1 else l for layer, l in node_keys]
                node_colors = [PRIMARY if layer == 0 else DECISION_COLORS.get(l, SLATE) if layer == 1 else band_color.get(l, SLATE) for layer, l in node_keys]
                sk = go.Figure(go.Sankey(arrangement="snap", node=dict(label=node_labels, color=node_colors, pad=22, thickness=18, line=dict(color="#fff", width=1)),
                                         link=dict(source=src, target=tgt, value=val, color=lcol)))
                show(sk, 400, legend=False)

            ab_df = fdf[fdf["Answerability"].map(is_answerable) & (fdf["Decision"] == "ABSTAIN")]
            ua_df = fdf[fdf["Answerability"].map(is_unanswerable) & (fdf["Decision"] == "ACCEPT")]
            n_answerable = int(fdf["Answerability"].map(is_answerable).sum())
            callout(f"{len(ab_df)} of {n_answerable} answerable questions in this view were routed to abstain, and {len(ua_df)} unanswerable "
                    "questions were accepted. These counts show where to look. They are not routing accuracy, because there are no independent human labels for the right decision.")

            c1, c2 = st.columns(2, gap="medium")
            with c1:
                with card("sun"):
                    ctitle("Question type, answerability, decision", "Click a segment to zoom in, click the centre to zoom out.")
                    sb = px.sunburst(fdf.assign(Decision_Label=fdf["Decision"].map(lambda d: DECISION_LABELS.get(d, d))),
                                     path=["Category", "Answerability", "Decision_Label"], color_discrete_sequence=[PRIMARY, "#6366F1", "#818CF8", "#A5B4FC", SLATE])
                    sb.update_traces(insidetextorientation="radial", marker=dict(line=dict(color="#fff", width=2)))
                    show(sb, 400, legend=False)
            with c2:
                with card("agree"):
                    ctitle("Agreement map", "Answerability on the rows, router decision on the columns.")
                    ct = pd.crosstab(fdf["Answerability"].astype(str), fdf["Decision"]).reindex(columns=DECISION_ORDER, fill_value=0)
                    hm = go.Figure(go.Heatmap(z=ct.values, x=[DECISION_LABELS[c] for c in ct.columns], y=list(ct.index),
                                              colorscale=[[0, "#F8FAFC"], [1, PRIMARY]], showscale=False, xgap=4, ygap=4,
                                              text=ct.values, texttemplate="%{text}", textfont=dict(size=22),
                                              hovertemplate="%{y}, %{x}: %{z} questions<extra></extra>"))
                    hm.update_yaxes(autorange="reversed")
                    show(hm, 400, legend=False)

            cfg = {"Faithfulness_Score": st.column_config.ProgressColumn("Grounding", min_value=0.0, max_value=1.0, format="%.2f")}
            w1, w2 = st.columns(2, gap="medium")
            with w1:
                with card("w1"):
                    ctitle("Answerable but abstained", "Possible false abstentions. Worth a manual read.")
                    if ab_df.empty:
                        st.caption("None in this view.")
                    else:
                        table(ab_df[["Question_ID", "Question", "Faithfulness_Score"]], column_config=cfg)
            with w2:
                with card("w2"):
                    ctitle("Unanswerable but accepted", "Possible unsafe accepts.")
                    if ua_df.empty:
                        st.caption("None in this view.")
                    else:
                        table(ua_df[["Question_ID", "Question", "Faithfulness_Score"]], column_config=cfg)

        else:  # Retrieval
            callout("These are development-set measurements. The exports contain no held-out retrieval score, so none is shown.")
            d_mrr = DEV_RETRIEVAL["MRR"]["improved"] - DEV_RETRIEVAL["MRR"]["baseline"]
            d_hit = DEV_RETRIEVAL["Hit@3"]["improved"] - DEV_RETRIEVAL["Hit@3"]["baseline"]
            kpi_cards([
                ("search", PRIMARY, "Hit@3 with reranking", f"{DEV_RETRIEVAL['Hit@3']['improved']:.3f}", f"{d_hit:+.3f} vs baseline",
                 "Share of questions whose gold evidence appears in the top three retrieved chunks.", DEV_RETRIEVAL["Hit@3"]["improved"]),
                ("target", VIOLET, "MRR with reranking", f"{DEV_RETRIEVAL['MRR']['improved']:.4f}", f"{d_mrr:+.4f} vs baseline",
                 "Mean reciprocal rank. Rewards putting the right chunk near the top.", DEV_RETRIEVAL["MRR"]["improved"]),
                ("database", CYAN, "Questions scored", f"{DEV_EVIDENCE_MAPPED}/{DEV_TOTAL}", "Have verified gold evidence",
                 "Development questions with verified gold evidence, the only ones retrieval can be scored on.", DEV_EVIDENCE_MAPPED / DEV_TOTAL),
            ])
            with card("dumbbell"):
                ctitle("Baseline against multi-query search with reranking", "Each row is one metric. The grey dot is the baseline and the indigo dot is the improved pipeline.")
                fig = go.Figure()
                for metric, v in DEV_RETRIEVAL.items():
                    fig.add_trace(go.Scatter(x=[v["baseline"], v["improved"]], y=[metric, metric], mode="lines", line=dict(color=LINE, width=6), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=[v["baseline"] for v in DEV_RETRIEVAL.values()], y=list(DEV_RETRIEVAL), mode="markers+text", name="Baseline",
                                         marker=dict(size=20, color=SLATE), text=[f"{v['baseline']:.3f}" for v in DEV_RETRIEVAL.values()], textposition="top center"))
                fig.add_trace(go.Scatter(x=[v["improved"] for v in DEV_RETRIEVAL.values()], y=list(DEV_RETRIEVAL), mode="markers+text", name="Multi-query + rerank",
                                         marker=dict(size=20, color=PRIMARY), text=[f"{v['improved']:.3f}" for v in DEV_RETRIEVAL.values()], textposition="bottom center"))
                fig.update_xaxes(range=[0, 0.6], title="Score")
                show(fig, 280)
            callout(f"Hit@3 did not move, so reranking found no additional questions whose evidence reached the top three. MRR rose by {d_mrr:.4f}, "
                    "which suggests relevant chunks tended to sit slightly higher. With 16 scored questions, one question changes Hit@3 by 0.0625, so treat small differences as noise.")

# =================================== EXPLORE A QUESTION ===================================
elif page == "Explore a question":
    heading("Explore a question", "Pick any held-out question to see the answer the system wrote, which claims were checked, and the evidence it used.")
    mode = seg("Show questions that were", ["All", "Accepted", "Sent to human review", "Abstained"], "qmode")
    pool = test_df if mode == "All" else test_df[test_df["Decision"] == {"Accepted": "ACCEPT", "Sent to human review": "HUMAN_REVIEW", "Abstained": "ABSTAIN"}[mode]]
    if pool.empty:
        st.info("No questions in this group. Choose another option above.")
    else:
        qids = pool["Question_ID"].astype(str).tolist()
        scored = pool[pool["Total_Claims"].fillna(0) > 0].sort_values("Faithfulness_Score")
        weak = str(scored.iloc[0]["Question_ID"]) if not scored.empty else qids[0]
        strong = str(scored.iloc[-1]["Question_ID"]) if not scored.empty else qids[0]
        if st.session_state.get("qsel") not in qids:
            st.session_state["qsel"] = qids[0]

        def pick(q):
            st.session_state["qsel"] = q

        b1, b2, b3, _ = st.columns([1.3, 1.5, 1.1, 3])
        b1.button("Show a weak answer", on_click=pick, args=(weak,), help="Lowest share of supported claims in this group.")
        b2.button("Show a strong answer", on_click=pick, args=(strong,), help="Highest share of supported claims in this group.")
        b3.button("Surprise me", on_click=pick, args=(pool.sample(1)["Question_ID"].astype(str).iloc[0],))
        selected = st.selectbox("Question", qids, key="qsel",
                                format_func=lambda q: f"{q}: {str(pool.loc[pool['Question_ID'].astype(str) == q, 'Question'].iloc[0])[:95]}")
        row = pool[pool["Question_ID"].astype(str) == selected].iloc[0]
        dcol = DECISION_COLORS.get(row["Decision"], SLATE)
        g = row["Faithfulness_Score"]
        kpi_cards([
            ("shield", dcol, "Router decision", DECISION_LABELS.get(row["Decision"], row["Decision"]), str(row["Answerability"]), "What the system decided to do with this answer.", None),
            ("check", GREEN, "Claim grounding", pct(g), f"{int(row['Supported_Claims'] or 0)} of {int(row['Total_Claims'] or 0)} claims supported",
             "Share of this answer's claims supported by evidence.", None if g is None or pd.isna(g) else g),
            ("layers", PRIMARY, "Question type", esc(row["Category"]), f"{len(row['Retrieved_Chunks'])} retrieved chunks", "The benchmark category of this question.", None),
            ("clock", CYAN, "Latency", f"{row['Latency_Seconds']:.2f}s" if pd.notna(row["Latency_Seconds"]) else "n/a", "Time to generate this answer", "Time taken to generate this answer.", None),
        ])
        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        st.markdown(f"**Question**  \n{row['Question'] or 'Question text not present in the export.'}")
        details = [c for c in row["Claim_Details"] if isinstance(c, dict)]
        if details:
            segs = "".join(f'<i style="--c:{[RED, AMBER, GREEN][verdict_code(c)]}" title="{esc(str(first(c, "claim", "Claim", "text", default=""))[:120])}"></i>' for c in details)
            st.markdown(f'<div class="csub" style="margin-top:12px">Claim strip: one segment per claim</div><div class="strip">{segs}</div>', unsafe_allow_html=True)
        body, located = xray(row["Generated_Answer"], row["Claim_Details"])
        st.markdown("**Generated answer, with claims highlighted**")
        st.markdown(f'<div class="xray">{body}</div>', unsafe_allow_html=True)
        if details:
            st.caption(f"{located} of {len(details)} claims were found word for word in the answer and highlighted. Green is supported, yellow neutral, red contradicted or unmatched.")

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        cl, ch = st.columns([1.15, 1], gap="large")
        with cl:
            st.markdown("**Claim by claim**")
            if not row["Claim_Details"]:
                st.info("No claim-level records were exported for this question.")
            for i, c in enumerate(row["Claim_Details"], 1):
                if not isinstance(c, dict):
                    st.write(str(c)); continue
                color, word = {2: (GREEN, "Supported"), 1: (AMBER, "Neutral"), 0: (RED, "Not supported")}[verdict_code(c)]
                sup_s = safe_float(first(c, "support_score", "Support_Score"))
                con_s = safe_float(first(c, "contradiction_score", "Contradiction_Score"))
                src_ = first(c, "supporting_source", "source", "Source", default="no source")
                pg = first(c, "supporting_page", "page", "Page", default="?")
                st.markdown(
                    f'<div class="claimrow" style="--c:{color};--t:{TXT[color]}"><b>Claim {i}: {word}</b>'
                    f'<div>{esc(first(c, "claim", "Claim", "text", default="(claim text unavailable)"))}</div>'
                    f'<small>Entailment {sup_s if sup_s is not None else "n/a"}, contradiction {con_s if con_s is not None else "n/a"}, '
                    f'{esc(src_)}, page {esc(pg)}</small></div>', unsafe_allow_html=True)
                ev = first(c, "supporting_evidence", "evidence", "Evidence", default="")
                if ev:
                    with st.expander(f"Evidence for claim {i}"):
                        st.write(ev)
        with ch:
            st.markdown("**Retrieved chunks**")
            if not row["Retrieved_Chunks"]:
                st.info("Retrieved chunks were not included in the exported answer results.")
            for idx, chunk in enumerate(row["Retrieved_Chunks"], 1):
                if not isinstance(chunk, dict):
                    chunk = {"content": str(chunk)}
                with st.expander(f"Rank {first(chunk, 'rank', 'Rank', default=idx)}: {first(chunk, 'source', 'Source', default='Unknown source')}, page {first(chunk, 'page', 'Page', default='?')}"):
                    st.write(first(chunk, "content", "text", "page_content", default="(No chunk text in export.)"))
        with st.expander("Reference answer"):
            st.write(row["Reference_Answer"] or "No reference answer in the export.")

# =================================== FINDINGS ===================================
else:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with card("limits"):
            ctitle("What these results do and do not show")
            items = [
                f"Held-out grounding is {pct(all_faith, 2)} ({all_sup} of {all_claims} claims) over {N} questions, a small benchmark.",
                "NLI scores screen claims. They are not calibrated probabilities that a claim is true.",
                "Sentence-level matching can miss claims that need several sentences of evidence.",
                "Zero questions were accepted on the test set, so the acceptance rule is very conservative." if A_ACC == 0
                else f"{A_ACC} questions were accepted on the test set. Check them for unsafe accepts.",
                f"{answerable_abstained} of {len(all_answerable)} answerable test questions were routed to abstain. Some may be false abstentions.",
                "Routing counts are not accuracy, because no independent human labels exist for the correct decision.",
                "Retrieval metrics come from the development set only. Thresholds were not tuned on the test set.",
            ]
            st.markdown("".join(f"- {i}\n" for i in items))

    heading("Glossary", "Plain-language definitions of the terms used in this dashboard.")
    gloss = pd.DataFrame([
        ("Claim", "One factual statement extracted from a generated answer."),
        ("Grounding (faithfulness)", "Share of an answer's claims that the NLI model found supported by the retrieved evidence."),
        ("NLI", "Natural language inference. A model that judges whether a passage supports, contradicts or says nothing about a statement."),
        ("Accepted", "The answer passed the acceptance rule and would be released automatically."),
        ("Sent to human review", "The answer is uncertain, so a person should read it first."),
        ("Abstained", "The system declined to answer."),
        ("Hit@3", "Share of questions whose correct source passage appears in the top three retrieved results."),
        ("MRR", "Mean reciprocal rank. Higher when the correct passage is ranked nearer the top."),
        ("Held-out test set", "Questions kept aside and never used to tune the system."),
    ], columns=["Term", "Meaning"])
    with card("gloss"):
        table(gloss)
