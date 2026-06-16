import os
import sys
import uuid
import streamlit as st

try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "⚠️ `GROQ_API_KEY` is missing from Streamlit secrets. "
        "Add it under your app's **Settings → Secrets** before continuing."
    )
    st.stop()

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from loader import load_docs
from chunker import split_docs
from embedder import EmbeddingManager
from vectore_store import VectorStoreManager
from retriever import RAGRetriever
from generator import generate_output

st.set_page_config(
    page_title="StudyMind AI — Workspace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Session State ──────────────────────────────────────────────────────────────
for key, val in [
    ("messages", []), ("retriever", None), ("vectorstore_ready", False),
    ("uploaded_file_names", []), ("quiz_data", None),
    ("active_tab", "chat"), ("workspace_open", False),
    ("session_id", None), ("vector_store", None),
    ("dark_mode", True),
]:
    if key not in st.session_state:
        st.session_state[key] = val

if st.session_state.session_id is None:
    st.session_state.session_id = uuid.uuid4().hex[:12]

if st.session_state.vector_store is None:
    st.session_state.vector_store = VectorStoreManager(persistent=False)

DATA_DIR = os.path.join("data", st.session_state.session_id)

# ── Theme tokens ───────────────────────────────────────────────────────────────
DARK = {
    "ACCENT":  "#f0b35a",
    "ACCENT2": "#f7c87a",
    "BG":      "#0e0e0e",
    "BG2":     "#161616",
    "BG3":     "#1e1e1e",
    "BORDER":  "#262626",
    "TEXT":    "#f5f0e8",
    "TEXT2":   "#8a8a8a",
}
LIGHT = {
    "ACCENT":  "#d4881e",
    "ACCENT2": "#e8a030",
    "BG":      "#f7f4ef",
    "BG2":     "#ffffff",
    "BG3":     "#ede9e2",
    "BORDER":  "#ddd9d0",
    "TEXT":    "#1a1714",
    "TEXT2":   "#6b6560",
}

T = DARK if st.session_state.dark_mode else LIGHT
ACCENT  = T["ACCENT"]
ACCENT2 = T["ACCENT2"]
BG      = T["BG"]
BG2     = T["BG2"]
BG3     = T["BG3"]
BORDER  = T["BORDER"]
TEXT    = T["TEXT"]
TEXT2   = T["TEXT2"]

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://api.fontshare.com/v2/css?f[]=clash-display@400,500,600,700&f[]=satoshi@300,400,500,700&display=swap');

html, body, [class*="css"] {{ font-family: 'Satoshi', sans-serif; }}
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stSidebar"] {{ display: none !important; }}
[data-testid="collapsedControl"] {{ display: none !important; }}

.stApp {{ background: {BG}; color: {TEXT}; }}

.stApp, .stApp p, .stApp span, .stApp div,
.stApp label, .stApp li, .stApp td, .stApp th,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] div {{
    color: {TEXT};
}}

h1,h2,h3 {{
    font-family: 'Clash Display', sans-serif !important;
    color: {TEXT} !important;
    font-weight: 700 !important;
}}

.hero-title {{ color: {TEXT} !important; }}
.hero-sub   {{ color: {TEXT2} !important; }}
.s-title    {{ color: {TEXT} !important; }}
.card-body  {{ color: {TEXT2} !important; }}
.stat-lbl   {{ color: {TEXT2} !important; }}
.feat-label {{ color: {TEXT2} !important; }}
.ws-logo    {{ color: {TEXT} !important; }}
.score-msg  {{ color: {TEXT2} !important; }}

/* ── Nav ── */
.ws-nav {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 0 28px;
    border-bottom: 1px solid {BORDER};
}}
.ws-logo {{
    font-family: 'Clash Display', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: {TEXT};
    display: flex;
    align-items: center;
    gap: 8px;
}}
.ws-dot {{
    width: 7px; height: 7px;
    background: {ACCENT};
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2.5s ease-in-out infinite;
}}
@keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1);}} 50%{{opacity:0.5;transform:scale(0.85);}} }}

/* ── Hero ── */
.hero-section {{
    padding: 80px 0 72px;
    border-bottom: 1px solid {BORDER};
    position: relative;
    overflow: hidden;
}}
.hero-bg {{
    position: absolute;
    top: 6px; right: -16px;
    font-family: 'Clash Display', sans-serif;
    font-size: clamp(3.5rem, 11vw, 8.5rem);
    font-weight: 700;
    color: {BG3};
    line-height: 1;
    pointer-events: none;
    user-select: none;
    opacity: 0.6;
    letter-spacing: 0.06em;
    white-space: nowrap;
}}
.hero-eye {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {ACCENT};
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 7px;
}}
.hero-title {{
    font-family: 'Clash Display', sans-serif;
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 700;
    letter-spacing: -0.035em;
    line-height: 1.15;
    color: {TEXT};
    margin-bottom: 20px;
}}
.hero-title em {{ color: {ACCENT}; font-style: italic; }}
.hero-sub {{
    color: {TEXT2};
    font-size: 0.95rem;
    line-height: 1.8;
    max-width: 480px;
    margin-bottom: 36px;
}}

/* ── Stats ── */
.stat-num {{
    font-family: 'Clash Display', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
    background: linear-gradient(90deg, {ACCENT} 0%, {ACCENT2} 40%, {ACCENT} 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 2.5s linear infinite;
}}
@keyframes shimmer {{
    0% {{ background-position: 200% center; }}
    100% {{ background-position: -200% center; }}
}}
.stats-row {{
    display: flex;
    gap: 0;
    border: 1px solid {BORDER};
    border-radius: 14px;
    overflow: hidden;
    margin-top: 56px;
}}
.stat-item {{
    flex: 1;
    padding: 20px 28px;
    border-right: 1px solid {BORDER};
    background: {BG2};
}}
.stat-item:last-child {{ border-right: none; }}
.stat-lbl {{ color: {TEXT2}; font-size: 0.75rem; }}

/* ── Upload section ── */
.upload-section {{
    padding: 64px 0 52px;
    border-bottom: 1px solid {BORDER};
}}
.s-eye {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {ACCENT};
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.s-title {{
    font-family: 'Clash Display', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: {TEXT};
    line-height: 1.05;
    margin-bottom: 28px;
}}
.s-title em {{ color: {ACCENT}; font-style: italic; }}

/* ── Cards ── */
.card {{
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 24px 28px;
}}
.card-title {{
    font-family: 'Clash Display', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    color: {ACCENT};
    margin-bottom: 12px;
}}
.card-body {{
    color: {TEXT2};
    font-size: 0.82rem;
    line-height: 2;
}}

/* ── Feature strip ── */
.feat-strip {{
    display: flex;
    gap: 1px;
    background: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
    margin-top: 16px;
}}
.feat-item {{
    flex: 1;
    background: {BG2};
    padding: 18px 20px;
    text-align: center;
}}
.feat-icon {{ font-size: 1.4rem; margin-bottom: 6px; }}
.feat-label {{ font-size: 0.74rem; color: {TEXT2}; font-weight: 500; }}

/* ── File badges ── */
.fbadge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: {BG3};
    color: {ACCENT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.76rem;
    font-weight: 500;
    margin: 3px 0;
    height: 34px;
    box-sizing: border-box;
}}
.fbadge-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}}
.fbadge-row .stButton > button {{
    background: {BG3} !important;
    color: {TEXT2} !important;
    border: 1px solid {BORDER} !important;
    padding: 4px 10px !important;
    font-size: 0.72rem !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}}
.fbadge-row .stButton > button:hover {{
    color: {ACCENT} !important;
    border-color: {ACCENT} !important;
    transform: none !important;
}}

/* ── Progress ── */
.progress-card {{
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 20px 24px;
    margin: 18px 0;
}}
.progress-step {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 0;
    font-size: 0.85rem;
    color: {TEXT2};
}}
.progress-step.done {{ color: {TEXT}; }}
.progress-step.active {{ color: {ACCENT}; font-weight: 600; }}
.progress-icon {{ width: 18px; text-align: center; font-size: 0.9rem; }}
.progress-step.active .progress-icon {{ animation: pulse 1.2s ease-in-out infinite; }}
.stProgress > div > div > div > div {{ background-color: {ACCENT} !important; }}
.stProgress > div > div {{ background-color: {BG3} !important; }}

/* ── Workspace ── */
.workspace-section {{ padding: 52px 0; }}

/* ── Chat ── */
[data-testid="stChatMessage"] {{
    background: {BG2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    margin: 6px 0 !important;
}}
[data-testid="stChatMessage"] * {{
    color: {TEXT} !important;
}}
[data-testid="stChatInput"] {{
    background: {BG2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
}}
[data-testid="stChatInput"]:focus-within {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 2px rgba(240,179,90,0.12) !important;
}}
[data-testid="stChatInput"] textarea {{ color: {TEXT} !important; }}
[data-testid="stChatInput"] textarea::placeholder {{ color: {TEXT2} !important; opacity: 0.8; }}

/* ── Inputs ── */
.stTextInput > div > div > input {{
    background: {BG2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 2px rgba(240,179,90,0.12) !important;
}}
.stSelectbox > div > div {{
    background: {BG2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
}}
[data-testid="stFileUploader"] {{
    background: {BG2} !important;
    border: 1.5px dashed {BORDER} !important;
    border-radius: 14px !important;
    padding: 8px !important;
}}
[data-testid="stFileUploader"] button {{
    background: {ACCENT} !important;
    color: {BG} !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {{
    color: {TEXT2} !important;
}}
[data-testid="stFileUploaderFile"] {{
    background: {BG3} !important;
    border-radius: 8px !important;
}}
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFile"] * {{
    color: {TEXT} !important;
    fill: {TEXT2} !important;
}}
[data-testid="stFileUploaderFile"] small {{
    color: {TEXT2} !important;
}}
[data-testid="stFileUploaderFile"]:hover svg {{
    fill: {ACCENT} !important;
}}
[data-testid="stFileUploaderFileData"], [data-testid="stFileUploaderFileData"] * {{
    color: {TEXT} !important;
}}

.stRadio > div {{
    background: {BG2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    margin: 3px 0 !important;
}}
.stAlert {{
    background: {BG2} !important;
    border-radius: 10px !important;
    border: 1px solid {BORDER} !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: {ACCENT} !important;
    color: {BG} !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Clash Display', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    padding: 10px 22px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em !important;
}}
.stButton > button:hover {{
    background: {ACCENT2} !important;
    transform: translateY(-2px) !important;
}}

.theme-toggle .stButton > button {{
    background: {BG3} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    padding: 6px 14px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    border-radius: 20px !important;
}}
.theme-toggle .stButton > button:hover {{
    border-color: {ACCENT} !important;
    color: {ACCENT} !important;
    transform: none !important;
}}

/* ── Status banner ── */
.status-banner {{
    display: flex;
    align-items: center;
    gap: 12px;
    background: {BG2};
    border: 1px solid {ACCENT};
    border-radius: 12px;
    padding: 14px 20px;
    margin-top: 14px;
}}
.status-banner svg {{ flex-shrink: 0; }}
.status-banner span {{
    font-size: 0.85rem;
    font-weight: 500;
    color: {TEXT} !important;
}}

/* ── Score ── */
.score-card {{
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 40px;
    text-align: center;
}}
.score-num {{
    font-family: 'Clash Display', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    color: {ACCENT};
    line-height: 1;
}}
.score-msg {{ color: {TEXT2}; font-size: 0.85rem; margin-top: 10px; }}

hr {{ border-color: {BORDER} !important; }}
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 2px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}

@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(22px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.a1 {{ animation: fadeUp 0.55s ease forwards; }}
.a2 {{ animation: fadeUp 0.55s 0.1s ease forwards; opacity:0; }}
.a3 {{ animation: fadeUp 0.55s 0.18s ease forwards; opacity:0; }}
.a4 {{ animation: fadeUp 0.55s 0.26s ease forwards; opacity:0; }}
.a5 {{ animation: fadeUp 0.55s 0.34s ease forwards; opacity:0; }}
.a6 {{ animation: fadeUp 0.55s 0.42s ease forwards; opacity:0; }}
</style>
""", unsafe_allow_html=True)

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚡ Loading embedding model... (first time only)")
def get_embedding_manager():
    return EmbeddingManager()

# Warm up the model immediately on startup so it's ready before the user clicks anything
get_embedding_manager()

# ── Status banner helper ───────────────────────────────────────────────────────
def render_status_banner(message):
    st.markdown(f"""
    <div class="status-banner">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{ACCENT}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
        <span>{message}</span>
    </div>
    """, unsafe_allow_html=True)

# ── Nav ───────────────────────────────────────────────────────────────────────
nav_left, nav_right = st.columns([6, 1])
with nav_left:
    st.markdown(f"""
    <div class="ws-nav a1">
        <div class="ws-logo" style="color:{TEXT};">
            <span class="ws-dot"></span>
            Study<span style="color:{ACCENT}">Mind</span> AI
        </div>
    </div>
    """, unsafe_allow_html=True)
with nav_right:
    st.markdown('<div class="theme-toggle" style="padding-top:18px;">', unsafe_allow_html=True)
    label = "Light mode" if st.session_state.dark_mode else "Dark mode"
    if st.button(label, key="theme_btn"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-section a2">
    <div class="hero-eye">● StudyMind AI · RAG Pipeline</div>
    <div class="hero-title">Your personal <em>study assistant.</em></div>
    <div class="hero-sub">
        Upload your PDFs, ask anything in plain English, and auto-generate quizzes —
        all powered by a local RAG pipeline. No hallucinations. Just your material.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="stats-row a3">
    <div class="stat-item"><div class="stat-num">384</div><div class="stat-lbl">Embedding dims</div></div>
    <div class="stat-item"><div class="stat-num">512</div><div class="stat-lbl">Tokens/chunk</div></div>
    <div class="stat-item"><div class="stat-num">∞</div><div class="stat-lbl">PDFs supported</div></div>
    <div class="stat-item"><div class="stat-num">Top 5</div><div class="stat-lbl">Retrieved chunks</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ── Upload Section ────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="upload-section">
    <div class="s-eye a4">● Step 01</div>
    <div class="s-title a4">Upload your <em>PDFs.</em></div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1], gap="large")
with col1:
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
with col2:
    st.markdown(f"""
    <div class="card a5">
        <div class="card-title">How it works</div>
        <div class="card-body">
            ① Upload your PDFs<br>
            ② Click Process PDFs<br>
            ③ Open Workspace<br>
            ④ Chat or take a quiz
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="feat-strip a6">
    <div class="feat-item"><div class="feat-icon">💬</div><div class="feat-label">Chat with PDFs</div></div>
    <div class="feat-item"><div class="feat-icon">🧩</div><div class="feat-label">Auto Quiz</div></div>
    <div class="feat-item"><div class="feat-icon">⚡</div><div class="feat-label">Local Embeddings</div></div>
    <div class="feat-item"><div class="feat-icon">🔗</div><div class="feat-label">Source Citations</div></div>
    <div class="feat-item"><div class="feat-icon">💾</div><div class="feat-label">Persistent Store</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Progress helper ───────────────────────────────────────────────────────────
PROCESS_STEPS = ["Saving files", "Loading PDFs", "Chunking text", "Generating embeddings", "Indexing"]

def render_progress(box, active_idx):
    html = '<div class="progress-card">'
    for i, step in enumerate(PROCESS_STEPS):
        if i < active_idx:
            icon, cls = "✓", "done"
        elif i == active_idx:
            icon, cls = "●", "active"
        else:
            icon, cls = "○", ""
        html += f'<div class="progress-step {cls}"><span class="progress-icon">{icon}</span>{step}</div>'
    html += '</div>'
    box.markdown(html, unsafe_allow_html=True)

# ── Process button ────────────────────────────────────────────────────────────
if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_file_names]

    if new_files:
        if st.button("Process PDFs"):
            embedding_manager = get_embedding_manager()
            vector_store = st.session_state.vector_store
            os.makedirs(DATA_DIR, exist_ok=True)
            progress_box = st.empty()
            progress_bar_slot = st.empty()

            try:
                render_progress(progress_box, 0)
                saved_paths = []
                for f in new_files:
                    safe_name = os.path.basename(f.name)
                    file_path = os.path.join(DATA_DIR, safe_name)
                    with open(file_path, "wb") as out:
                        out.write(f.read())
                    saved_paths.append(file_path)

                render_progress(progress_box, 1)
                docs = load_docs(file_paths=saved_paths)
                if not docs:
                    raise ValueError("No readable pages were found in the uploaded PDF(s).")

                render_progress(progress_box, 2)
                chunks = split_docs(docs)

                render_progress(progress_box, 3)
                progress_bar = progress_bar_slot.progress(0)

                def update_progress(done, total):
                    progress_bar.progress(done / total)

                embeddings = embedding_manager.generate_embedding(
                    [d.page_content for d in chunks],
                    progress_callback=update_progress
                )
                progress_bar_slot.empty()

                render_progress(progress_box, 4)
                vector_store.add_documents(chunks, embeddings, replace=False)
                render_progress(progress_box, len(PROCESS_STEPS))

                st.session_state.retriever = RAGRetriever(embedding_manager, vector_store)
                st.session_state.vectorstore_ready = True
                st.session_state.uploaded_file_names.extend([f.name for f in new_files])
                st.session_state.workspace_open = True

                progress_box.empty()
                count = len(new_files)
                render_status_banner(f"{count} PDF{'s' if count != 1 else ''} processed and ready")

            except Exception as e:
                progress_box.empty()
                progress_bar_slot.empty()
                st.error(f"Something went wrong while processing your PDF(s): {e}")
    else:
        render_status_banner(f"All {len(uploaded_files)} PDF{'s' if len(uploaded_files) != 1 else ''} already loaded")
        if not st.session_state.workspace_open:
            if st.button("Open Workspace"):
                st.session_state.workspace_open = True
                st.rerun()

# ── File badges ───────────────────────────────────────────────────────────────
if st.session_state.vectorstore_ready and st.session_state.uploaded_file_names:
    st.markdown('<div style="margin-top:14px;">', unsafe_allow_html=True)
    for fname in list(st.session_state.uploaded_file_names):
        bcol1, bcol2 = st.columns([6, 1])
        with bcol1:
            st.markdown(f'<div class="fbadge">📄 {fname}</div>', unsafe_allow_html=True)
        with bcol2:
            st.markdown('<div class="fbadge-row">', unsafe_allow_html=True)
            if st.button("Remove", key=f"remove_{fname}"):
                file_path = os.path.join(DATA_DIR, fname)
                st.session_state.vector_store.delete_by_source(file_path)
                st.session_state.uploaded_file_names.remove(fname)
                if os.path.exists(file_path):
                    os.remove(file_path)
                if not st.session_state.uploaded_file_names:
                    st.session_state.vectorstore_ready = False
                    st.session_state.workspace_open = False
                    st.session_state.messages = []
                    st.session_state.quiz_data = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Workspace ─────────────────────────────────────────────────────────────────
if not st.session_state.workspace_open or not st.session_state.vectorstore_ready:
    st.stop()

st.markdown(f'<div style="border-top:1px solid {BORDER};margin:48px 0 0;"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="workspace-section">
    <div class="s-eye">● Step 02</div>
    <div class="s-title">Chat or <em>test yourself.</em></div>
</div>
""", unsafe_allow_html=True)

col_c, col_q, col_clear, col_sp = st.columns([1, 1, 1, 5])
with col_c:
    if st.button("Chat", key="btn_chat"):
        st.session_state.active_tab = "chat"
        st.rerun()
with col_q:
    if st.button("Quiz", key="btn_quiz"):
        st.session_state.active_tab = "quiz"
        st.rerun()
with col_clear:
    if st.button("Clear", key="btn_clear"):
        st.session_state.messages = []
        st.session_state.quiz_data = None
        st.rerun()

st.markdown(f'<div style="border-bottom:1px solid {BORDER};margin:8px 0 28px;"></div>', unsafe_allow_html=True)

# ── Chat ──────────────────────────────────────────────────────────────────────
if st.session_state.active_tab == "chat":
    col_main, col_side = st.columns([4, 1], gap="large")

    with col_side:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Tips</div>
            <div class="card-body">
                Ask specific questions<br>
                Reference topic names<br>
                Ask for summaries<br>
                Compare concepts
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_main:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        if prompt := st.chat_input("Ask anything from your PDFs..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = generate_output(prompt, st.session_state.retriever)
                    except Exception as e:
                        answer = f"Sorry, something went wrong generating a response: {e}"
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Quiz ──────────────────────────────────────────────────────────────────────
elif st.session_state.active_tab == "quiz":
    col_form, col_tip = st.columns([3, 1], gap="large")

    with col_tip:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Quiz Tips</div>
            <div class="card-body">
                Leave blank for general<br>
                Specify topic to focus<br>
                Start with 3 questions
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_form:
        c1, c2 = st.columns([3, 1])
        with c1:
            topic = st.text_input("Topic (optional)", placeholder="e.g. Transformers, RAG, Attention...")
        with c2:
            num_q = st.selectbox("Questions", [3, 5, 10])

        if st.button("Generate Quiz"):
            with st.spinner("Generating..."):
                try:
                    from quiz import generate_quiz
                    st.session_state.quiz_data = generate_quiz(
                        retriever=st.session_state.retriever,
                        topic=topic,
                        num_questions=num_q
                    )
                except Exception as e:
                    st.session_state.quiz_data = None
                    st.error(f"Couldn't generate a quiz: {e}")

    if st.session_state.quiz_data:
        st.markdown(f'<div style="border-top:1px solid {BORDER};margin:24px 0;"></div>', unsafe_allow_html=True)
        answers = {}
        for i, q in enumerate(st.session_state.quiz_data):
            st.markdown('<div class="card" style="margin:10px 0;">', unsafe_allow_html=True)
            st.markdown(f"**Q{i+1}. {q['question']}**")
            answers[i] = st.radio("", q["options"], key=f"q{i}", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Submit Quiz"):
            score = sum(1 for i, q in enumerate(st.session_state.quiz_data) if answers[i] == q["answer"])
            total = len(st.session_state.quiz_data)
            msg = "Perfect score!" if score == total else "Good job! Keep revising." if score >= total//2 else "Need more practice."
            st.markdown(f"""
            <div class="score-card">
                <div class="score-num">{score} / {total}</div>
                <div class="score-msg">{msg}</div>
            </div>
            """, unsafe_allow_html=True)
            if score == total:
                st.balloons()