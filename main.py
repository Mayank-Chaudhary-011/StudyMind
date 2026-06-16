import os
import sys
import streamlit as st

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
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

# ── Color Scheme: Deep charcoal + warm coral ───────────────────────────────────
ACCENT  = "#f0b35a"
ACCENT2 = "#f0b35a"
BG      = "#0e0e0e"
BG2     = "#161616"
BG3     = "#1e1e1e"
BORDER  = "#262626"
TEXT    = "#f5f0e8"
TEXT2   = "#5a5a5a"
TEXT3   = "#3a3a3a"

st.markdown(f"""
<style>
@import url('https://api.fontshare.com/v2/css?f[]=clash-display@400,500,600,700&f[]=satoshi@300,400,500,700&display=swap');

/* ── Force dark mode always ── */
:root {{
    color-scheme: dark !important;
}}
html, body {{
    background-color: #0e0e0e !important;
    color: #f5f0e8 !important;
}}
/* Force all Streamlit text to light color */
.stApp, .stApp * {{
    color: #f5f0e8 !important;
}}
/* Fix chat message text specifically */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {{
    color: #f5f0e8 !important;
}}
/* Fix markdown text */
.stMarkdown p,
.stMarkdown span,
.element-container p {{
    color: #f5f0e8 !important;
}}
/* Fix input placeholder */
textarea::placeholder,
input::placeholder {{
    color: #5a5a5a !important;
}}

html, body, [class*="css"] {{ font-family: 'Satoshi', sans-serif; }}
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stSidebar"] {{ display: none !important; }}
[data-testid="collapsedControl"] {{ display: none !important; }}

/* ── Floating Particles ── */
.pt{position:fixed;border-radius:50%;background:#f0b35a;pointer-events:none;z-index:1;}
.pt0{left:52vw;top:49vh;width:2.6px;height:2.6px;opacity:0.29;animation:flt0 10.1s 3.0s ease-in-out infinite alternate;}
@keyframes flt0{0%{transform:translate(0,0);opacity:0.29;}100%{transform:translate(74px,-98px);opacity:0.09;}}
.pt1{left:50vw;top:68vh;width:4.1px;height:4.1px;opacity:0.35;animation:flt1 14.5s 11.3s ease-in-out infinite alternate;}
@keyframes flt1{0%{transform:translate(0,0);opacity:0.35;}100%{transform:translate(5px,-69px);opacity:0.1;}}
.pt2{left:79vw;top:76vh;width:2.7px;height:2.7px;opacity:0.44;animation:flt2 18.8s 4.5s ease-in-out infinite alternate;}
@keyframes flt2{0%{transform:translate(0,0);opacity:0.44;}100%{transform:translate(-21px,100px);opacity:0.13;}}
.pt3{left:20vw;top:60vh;width:4.0px;height:4.0px;opacity:0.44;animation:flt3 18.2s 4.1s ease-in-out infinite alternate;}
@keyframes flt3{0%{transform:translate(0,0);opacity:0.44;}100%{transform:translate(63px,15px);opacity:0.13;}}
.pt4{left:12vw;top:61vh;width:2.9px;height:2.9px;opacity:0.33;animation:flt4 8.5s 6.7s ease-in-out infinite alternate;}
@keyframes flt4{0%{transform:translate(0,0);opacity:0.33;}100%{transform:translate(-12px,-68px);opacity:0.1;}}
.pt5{left:69vw;top:11vh;width:3.4px;height:3.4px;opacity:0.25;animation:flt5 18.8s 7.3s ease-in-out infinite alternate;}
@keyframes flt5{0%{transform:translate(0,0);opacity:0.25;}100%{transform:translate(-107px,-29px);opacity:0.07;}}
.pt6{left:92vw;top:21vh;width:4.2px;height:4.2px;opacity:0.43;animation:flt6 13.3s 11.1s ease-in-out infinite alternate;}
@keyframes flt6{0%{transform:translate(0,0);opacity:0.43;}100%{transform:translate(-75px,8px);opacity:0.13;}}
.pt7{left:5vw;top:15vh;width:2.1px;height:2.1px;opacity:0.35;animation:flt7 13.7s 9.8s ease-in-out infinite alternate;}
@keyframes flt7{0%{transform:translate(0,0);opacity:0.35;}100%{transform:translate(-100px,119px);opacity:0.1;}}
.pt8{left:12vw;top:19vh;width:2.9px;height:2.9px;opacity:0.27;animation:flt8 8.1s 4.0s ease-in-out infinite alternate;}
@keyframes flt8{0%{transform:translate(0,0);opacity:0.27;}100%{transform:translate(13px,-17px);opacity:0.08;}}
.pt9{left:8vw;top:82vh;width:2.6px;height:2.6px;opacity:0.32;animation:flt9 11.1s 4.8s ease-in-out infinite alternate;}
@keyframes flt9{0%{transform:translate(0,0);opacity:0.32;}100%{transform:translate(75px,-88px);opacity:0.1;}}
.pt10{left:89vw;top:61vh;width:4.8px;height:4.8px;opacity:0.46;animation:flt10 9.6s 10.4s ease-in-out infinite alternate;}
@keyframes flt10{0%{transform:translate(0,0);opacity:0.46;}100%{transform:translate(33px,79px);opacity:0.14;}}
.pt11{left:78vw;top:53vh;width:2.4px;height:2.4px;opacity:0.25;animation:flt11 9.4s 3.6s ease-in-out infinite alternate;}
@keyframes flt11{0%{transform:translate(0,0);opacity:0.25;}100%{transform:translate(-75px,-16px);opacity:0.07;}}
.pt12{left:27vw;top:72vh;width:3.0px;height:3.0px;opacity:0.5;animation:flt12 14.0s 2.8s ease-in-out infinite alternate;}
@keyframes flt12{0%{transform:translate(0,0);opacity:0.5;}100%{transform:translate(-55px,-90px);opacity:0.15;}}
.pt13{left:47vw;top:24vh;width:4.0px;height:4.0px;opacity:0.32;animation:flt13 14.0s 0.1s ease-in-out infinite alternate;}
@keyframes flt13{0%{transform:translate(0,0);opacity:0.32;}100%{transform:translate(93px,-108px);opacity:0.1;}}
.pt14{left:49vw;top:52vh;width:3.0px;height:3.0px;opacity:0.39;animation:flt14 17.7s 4.6s ease-in-out infinite alternate;}
@keyframes flt14{0%{transform:translate(0,0);opacity:0.39;}100%{transform:translate(88px,-26px);opacity:0.12;}}
.pt15{left:24vw;top:5vh;width:4.8px;height:4.8px;opacity:0.31;animation:flt15 16.7s 4.8s ease-in-out infinite alternate;}
@keyframes flt15{0%{transform:translate(0,0);opacity:0.31;}100%{transform:translate(-7px,-62px);opacity:0.09;}}
.pt16{left:84vw;top:29vh;width:4.0px;height:4.0px;opacity:0.28;animation:flt16 11.7s 10.5s ease-in-out infinite alternate;}
@keyframes flt16{0%{transform:translate(0,0);opacity:0.28;}100%{transform:translate(64px,-40px);opacity:0.08;}}
.pt17{left:92vw;top:87vh;width:4.1px;height:4.1px;opacity:0.35;animation:flt17 12.6s 5.3s ease-in-out infinite alternate;}
@keyframes flt17{0%{transform:translate(0,0);opacity:0.35;}100%{transform:translate(-100px,105px);opacity:0.1;}}
.pt18{left:33vw;top:9vh;width:3.5px;height:3.5px;opacity:0.39;animation:flt18 18.9s 10.5s ease-in-out infinite alternate;}
@keyframes flt18{0%{transform:translate(0,0);opacity:0.39;}100%{transform:translate(81px,2px);opacity:0.12;}}
.pt19{left:33vw;top:56vh;width:4.6px;height:4.6px;opacity:0.3;animation:flt19 12.9s 7.9s ease-in-out infinite alternate;}
@keyframes flt19{0%{transform:translate(0,0);opacity:0.3;}100%{transform:translate(36px,-80px);opacity:0.09;}}
.pt20{left:87vw;top:87vh;width:3.8px;height:3.8px;opacity:0.33;animation:flt20 19.8s 8.0s ease-in-out infinite alternate;}
@keyframes flt20{0%{transform:translate(0,0);opacity:0.33;}100%{transform:translate(-8px,-71px);opacity:0.1;}}
.pt21{left:26vw;top:45vh;width:2.0px;height:2.0px;opacity:0.51;animation:flt21 12.3s 3.5s ease-in-out infinite alternate;}
@keyframes flt21{0%{transform:translate(0,0);opacity:0.51;}100%{transform:translate(97px,-71px);opacity:0.15;}}
.pt22{left:86vw;top:22vh;width:4.4px;height:4.4px;opacity:0.45;animation:flt22 11.1s 5.0s ease-in-out infinite alternate;}
@keyframes flt22{0%{transform:translate(0,0);opacity:0.45;}100%{transform:translate(72px,101px);opacity:0.14;}}
.pt23{left:53vw;top:45vh;width:4.0px;height:4.0px;opacity:0.38;animation:flt23 8.0s 6.0s ease-in-out infinite alternate;}
@keyframes flt23{0%{transform:translate(0,0);opacity:0.38;}100%{transform:translate(-69px,-120px);opacity:0.11;}}
.pt24{left:87vw;top:96vh;width:5.0px;height:5.0px;opacity:0.29;animation:flt24 14.1s 9.7s ease-in-out infinite alternate;}
@keyframes flt24{0%{transform:translate(0,0);opacity:0.29;}100%{transform:translate(118px,-68px);opacity:0.09;}}
.pt25{left:95vw;top:54vh;width:3.4px;height:3.4px;opacity:0.47;animation:flt25 14.4s 7.4s ease-in-out infinite alternate;}
@keyframes flt25{0%{transform:translate(0,0);opacity:0.47;}100%{transform:translate(41px,92px);opacity:0.14;}}
.pt26{left:99vw;top:37vh;width:2.6px;height:2.6px;opacity:0.4;animation:flt26 8.2s 1.3s ease-in-out infinite alternate;}
@keyframes flt26{0%{transform:translate(0,0);opacity:0.4;}100%{transform:translate(-42px,-101px);opacity:0.12;}}
.pt27{left:8vw;top:54vh;width:2.9px;height:2.9px;opacity:0.4;animation:flt27 19.6s 7.5s ease-in-out infinite alternate;}
@keyframes flt27{0%{transform:translate(0,0);opacity:0.4;}100%{transform:translate(61px,30px);opacity:0.12;}}
.pt28{left:25vw;top:97vh;width:2.3px;height:2.3px;opacity:0.29;animation:flt28 18.4s 4.7s ease-in-out infinite alternate;}
@keyframes flt28{0%{transform:translate(0,0);opacity:0.29;}100%{transform:translate(82px,-76px);opacity:0.09;}}
.pt29{left:43vw;top:17vh;width:4.6px;height:4.6px;opacity:0.34;animation:flt29 18.8s 10.4s ease-in-out infinite alternate;}
@keyframes flt29{0%{transform:translate(0,0);opacity:0.34;}100%{transform:translate(57px,59px);opacity:0.1;}}

/* Grain overlay */
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 9998;
    opacity: 0.04;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    background-size: 180px;
}}
.stApp {{ background: {BG}; color: {TEXT}; }}

h1,h2,h3 {{
    font-family: 'Clash Display', sans-serif !important;
    color: {TEXT} !important;
    font-weight: 700 !important;
}}

/* ── Nav ── */
.ws-nav {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 0 28px;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 0;
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

/* ── Hero section ── */
.hero-section {{
    padding: 80px 0 72px;
    border-bottom: 1px solid {BORDER};
    position: relative;
    overflow: hidden;
}}
.hero-bg {{
    position: absolute;
    top: -60px; right: -40px;
    font-family: 'Clash Display', sans-serif;
    font-size: 22rem;
    font-weight: 700;
    color: {BG3};
    line-height: 1;
    pointer-events: none;
    user-select: none;
    opacity: 0.5;
    letter-spacing: -0.05em;
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

/* ── Typing animation on hero title ── */
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
.typing-line {{
    display: block;
    overflow: hidden;
    white-space: nowrap;
    width: 0;
    border-right: 3px solid {ACCENT};
}}
.typing-line-1 {{
    animation: typing 1.2s steps(16, end) 0.3s forwards,
               blink 0.75s step-end 0.3s 2;
}}
.typing-line-2 {{
    animation: typing 1.4s steps(18, end) 1.6s forwards,
               blink 0.75s step-end infinite 1.6s;
}}
@keyframes typing {{
    from {{ width: 0; }}
    to {{ width: 100%; }}
}}
@keyframes blink {{
    0%, 100% {{ border-color: {ACCENT}; }}
    50% {{ border-color: transparent; }}
}}

.hero-sub {{
    color: {TEXT2};
    font-size: 0.95rem;
    line-height: 1.8;
    max-width: 480px;
    margin-bottom: 36px;
}}

/* ── Shimmer effect on stat numbers ── */
.stat-num {{
    font-family: 'Clash Display', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
    background: linear-gradient(
        90deg,
        {ACCENT} 0%,
        #fff8ee 40%,
        {ACCENT} 60%,
        {ACCENT} 100%
    );
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

/* Stats */
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

/* Cards */
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

/* Feature strip */
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

/* File badge */
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
    margin: 3px;
}}

/* ── Workspace ── */
.workspace-section {{ padding: 52px 0; }}
.tab-switcher {{
    display: flex;
    gap: 6px;
    margin-bottom: 28px;
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 4px;
    width: fit-content;
}}
.tab-btn {{
    padding: 8px 24px;
    border-radius: 7px;
    font-family: 'Clash Display', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: all 0.2s;
    letter-spacing: 0.02em;
}}
.tab-active {{ background: {ACCENT}; color: {BG}; }}
.tab-inactive {{ background: transparent; color: {TEXT2}; }}

/* Chat bubbles */
[data-testid="stChatMessage"] {{
    background: {BG2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    margin: 6px 0 !important;
}}
[data-testid="stChatInput"] {{
    background: {BG2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
}}
[data-testid="stChatInput"]:focus-within {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 2px rgba(255,123,84,0.12) !important;
}}
[data-testid="stChatInput"] textarea {{ color: {TEXT} !important; }}

/* Inputs */
.stTextInput > div > div > input {{
    background: {BG2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 2px rgba(255,123,84,0.12) !important;
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

/* Buttons */
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

/* Score */
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

/* Animations */
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

# ── Particles ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pt pt0"></div>
<div class="pt pt1"></div>
<div class="pt pt2"></div>
<div class="pt pt3"></div>
<div class="pt pt4"></div>
<div class="pt pt5"></div>
<div class="pt pt6"></div>
<div class="pt pt7"></div>
<div class="pt pt8"></div>
<div class="pt pt9"></div>
<div class="pt pt10"></div>
<div class="pt pt11"></div>
<div class="pt pt12"></div>
<div class="pt pt13"></div>
<div class="pt pt14"></div>
<div class="pt pt15"></div>
<div class="pt pt16"></div>
<div class="pt pt17"></div>
<div class="pt pt18"></div>
<div class="pt pt19"></div>
<div class="pt pt20"></div>
<div class="pt pt21"></div>
<div class="pt pt22"></div>
<div class="pt pt23"></div>
<div class="pt pt24"></div>
<div class="pt pt25"></div>
<div class="pt pt26"></div>
<div class="pt pt27"></div>
<div class="pt pt28"></div>
<div class="pt pt29"></div>
""", unsafe_allow_html=True)

# ── GSAP + Floating Particles ──────────────────────────────────────────────────
st.markdown("""<style>
/* Make particles escape iframe */
.particle { position: fixed !important; }

.particle{position:fixed !important;border-radius:50%;background:#f0b35a;pointer-events:none;z-index:9999 !important;}
.p0{left:83%;top:16%;width:2.1px;height:2.1px;opacity:0.46;animation:fl0 7.8s 2.2s ease-in-out infinite;}
@keyframes fl0{0%,100%{transform:translate(0,0);opacity:0.46;}50%{transform:translate(73px,89px);opacity:0.14;}}
.p1{left:71%;top:13%;width:3.8px;height:3.8px;opacity:0.28;animation:fl1 6.7s 0.3s ease-in-out infinite;}
@keyframes fl1{0%,100%{transform:translate(0,0);opacity:0.28;}50%{transform:translate(54px,-94px);opacity:0.08;}}
.p2{left:73%;top:27%;width:4.1px;height:4.1px;opacity:0.36;animation:fl2 9.4s 5.6s ease-in-out infinite;}
@keyframes fl2{0%,100%{transform:translate(0,0);opacity:0.36;}50%{transform:translate(-29px,-99px);opacity:0.11;}}
.p3{left:22%;top:91%;width:3.3px;height:3.3px;opacity:0.47;animation:fl3 7.7s 2.2s ease-in-out infinite;}
@keyframes fl3{0%,100%{transform:translate(0,0);opacity:0.47;}50%{transform:translate(-74px,-77px);opacity:0.14;}}
.p4{left:50%;top:14%;width:3.1px;height:3.1px;opacity:0.22;animation:fl4 8.1s 2.8s ease-in-out infinite;}
@keyframes fl4{0%,100%{transform:translate(0,0);opacity:0.22;}50%{transform:translate(17px,37px);opacity:0.07;}}
.p5{left:17%;top:50%;width:2.2px;height:2.2px;opacity:0.51;animation:fl5 11.0s 2.3s ease-in-out infinite;}
@keyframes fl5{0%,100%{transform:translate(0,0);opacity:0.51;}50%{transform:translate(-8px,47px);opacity:0.15;}}
.p6{left:26%;top:92%;width:2.2px;height:2.2px;opacity:0.54;animation:fl6 12.2s 5.3s ease-in-out infinite;}
@keyframes fl6{0%,100%{transform:translate(0,0);opacity:0.54;}50%{transform:translate(-41px,-75px);opacity:0.16;}}
.p7{left:50%;top:37%;width:3.4px;height:3.4px;opacity:0.32;animation:fl7 7.3s 6.7s ease-in-out infinite;}
@keyframes fl7{0%,100%{transform:translate(0,0);opacity:0.32;}50%{transform:translate(71px,-32px);opacity:0.1;}}
.p8{left:91%;top:89%;width:3.9px;height:3.9px;opacity:0.46;animation:fl8 7.4s 4.9s ease-in-out infinite;}
@keyframes fl8{0%,100%{transform:translate(0,0);opacity:0.46;}50%{transform:translate(-59px,18px);opacity:0.14;}}
.p9{left:50%;top:36%;width:5.0px;height:5.0px;opacity:0.44;animation:fl9 10.5s 5.1s ease-in-out infinite;}
@keyframes fl9{0%,100%{transform:translate(0,0);opacity:0.44;}50%{transform:translate(96px,98px);opacity:0.13;}}
.p10{left:9%;top:31%;width:4.5px;height:4.5px;opacity:0.22;animation:fl10 9.2s 6.4s ease-in-out infinite;}
@keyframes fl10{0%,100%{transform:translate(0,0);opacity:0.22;}50%{transform:translate(45px,83px);opacity:0.07;}}
.p11{left:42%;top:29%;width:4.0px;height:4.0px;opacity:0.36;animation:fl11 13.3s 3.2s ease-in-out infinite;}
@keyframes fl11{0%,100%{transform:translate(0,0);opacity:0.36;}50%{transform:translate(-33px,-65px);opacity:0.11;}}
.p12{left:33%;top:97%;width:3.7px;height:3.7px;opacity:0.51;animation:fl12 10.7s 2.1s ease-in-out infinite;}
@keyframes fl12{0%,100%{transform:translate(0,0);opacity:0.51;}50%{transform:translate(2px,-8px);opacity:0.15;}}
.p13{left:30%;top:19%;width:3.5px;height:3.5px;opacity:0.24;animation:fl13 6.4s 0.7s ease-in-out infinite;}
@keyframes fl13{0%,100%{transform:translate(0,0);opacity:0.24;}50%{transform:translate(60px,-60px);opacity:0.07;}}
.p14{left:89%;top:56%;width:3.8px;height:3.8px;opacity:0.36;animation:fl14 10.8s 3.1s ease-in-out infinite;}
@keyframes fl14{0%,100%{transform:translate(0,0);opacity:0.36;}50%{transform:translate(-36px,41px);opacity:0.11;}}
.p15{left:3%;top:89%;width:4.2px;height:4.2px;opacity:0.29;animation:fl15 10.3s 5.5s ease-in-out infinite;}
@keyframes fl15{0%,100%{transform:translate(0,0);opacity:0.29;}50%{transform:translate(64px,-13px);opacity:0.09;}}
.p16{left:16%;top:39%;width:3.3px;height:3.3px;opacity:0.51;animation:fl16 13.6s 3.6s ease-in-out infinite;}
@keyframes fl16{0%,100%{transform:translate(0,0);opacity:0.51;}50%{transform:translate(-33px,28px);opacity:0.15;}}
.p17{left:24%;top:66%;width:4.7px;height:4.7px;opacity:0.42;animation:fl17 8.4s 7.0s ease-in-out infinite;}
@keyframes fl17{0%,100%{transform:translate(0,0);opacity:0.42;}50%{transform:translate(55px,-50px);opacity:0.13;}}
.p18{left:21%;top:49%;width:4.3px;height:4.3px;opacity:0.39;animation:fl18 12.2s 4.3s ease-in-out infinite;}
@keyframes fl18{0%,100%{transform:translate(0,0);opacity:0.39;}50%{transform:translate(-100px,53px);opacity:0.12;}}
.p19{left:43%;top:64%;width:2.1px;height:2.1px;opacity:0.49;animation:fl19 13.0s 7.4s ease-in-out infinite;}
@keyframes fl19{0%,100%{transform:translate(0,0);opacity:0.49;}50%{transform:translate(-22px,-39px);opacity:0.15;}}
.p20{left:9%;top:32%;width:4.6px;height:4.6px;opacity:0.37;animation:fl20 6.7s 7.6s ease-in-out infinite;}
@keyframes fl20{0%,100%{transform:translate(0,0);opacity:0.37;}50%{transform:translate(-83px,94px);opacity:0.11;}}
.p21{left:70%;top:18%;width:2.4px;height:2.4px;opacity:0.29;animation:fl21 10.4s 3.8s ease-in-out infinite;}
@keyframes fl21{0%,100%{transform:translate(0,0);opacity:0.29;}50%{transform:translate(55px,8px);opacity:0.09;}}
.p22{left:29%;top:71%;width:4.3px;height:4.3px;opacity:0.34;animation:fl22 11.7s 5.5s ease-in-out infinite;}
@keyframes fl22{0%,100%{transform:translate(0,0);opacity:0.34;}50%{transform:translate(71px,66px);opacity:0.1;}}
.p23{left:49%;top:58%;width:4.7px;height:4.7px;opacity:0.22;animation:fl23 8.0s 3.6s ease-in-out infinite;}
@keyframes fl23{0%,100%{transform:translate(0,0);opacity:0.22;}50%{transform:translate(-95px,50px);opacity:0.07;}}
.p24{left:72%;top:31%;width:3.8px;height:3.8px;opacity:0.22;animation:fl24 11.7s 0.1s ease-in-out infinite;}
@keyframes fl24{0%,100%{transform:translate(0,0);opacity:0.22;}50%{transform:translate(-83px,-92px);opacity:0.07;}}
.p25{left:44%;top:11%;width:3.5px;height:3.5px;opacity:0.39;animation:fl25 9.9s 2.2s ease-in-out infinite;}
@keyframes fl25{0%,100%{transform:translate(0,0);opacity:0.39;}50%{transform:translate(85px,46px);opacity:0.12;}}
.p26{left:75%;top:62%;width:2.7px;height:2.7px;opacity:0.23;animation:fl26 9.3s 3.8s ease-in-out infinite;}
@keyframes fl26{0%,100%{transform:translate(0,0);opacity:0.23;}50%{transform:translate(68px,10px);opacity:0.07;}}
.p27{left:47%;top:56%;width:3.2px;height:3.2px;opacity:0.43;animation:fl27 6.4s 6.9s ease-in-out infinite;}
@keyframes fl27{0%,100%{transform:translate(0,0);opacity:0.43;}50%{transform:translate(65px,-75px);opacity:0.13;}}
.p28{left:9%;top:53%;width:4.2px;height:4.2px;opacity:0.27;animation:fl28 6.9s 6.4s ease-in-out infinite;}
@keyframes fl28{0%,100%{transform:translate(0,0);opacity:0.27;}50%{transform:translate(37px,14px);opacity:0.08;}}
.p29{left:19%;top:56%;width:2.6px;height:2.6px;opacity:0.23;animation:fl29 13.0s 3.7s ease-in-out infinite;}
@keyframes fl29{0%,100%{transform:translate(0,0);opacity:0.23;}50%{transform:translate(40px,-75px);opacity:0.07;}}

</style>
<div class="particle p0"></div><div class="particle p1"></div><div class="particle p2"></div><div class="particle p3"></div><div class="particle p4"></div><div class="particle p5"></div><div class="particle p6"></div><div class="particle p7"></div><div class="particle p8"></div><div class="particle p9"></div><div class="particle p10"></div><div class="particle p11"></div><div class="particle p12"></div><div class="particle p13"></div><div class="particle p14"></div><div class="particle p15"></div><div class="particle p16"></div><div class="particle p17"></div><div class="particle p18"></div><div class="particle p19"></div><div class="particle p20"></div><div class="particle p21"></div><div class="particle p22"></div><div class="particle p23"></div><div class="particle p24"></div><div class="particle p25"></div><div class="particle p26"></div><div class="particle p27"></div><div class="particle p28"></div><div class="particle p29"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js"></script>
<script>
window.addEventListener('load', function() {
    if (typeof gsap === 'undefined') return;
    gsap.registerPlugin(ScrollTrigger);
    gsap.utils.toArray('.card, .feat-item, .stat-item').forEach((el, i) => {
        gsap.fromTo(el,
            { opacity: 0, y: 24 },
            { opacity: 1, y: 0, duration: 0.5, delay: i * 0.06,
              scrollTrigger: { trigger: el, start: 'top 90%' } }
        );
    });
});
</script>""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
for key, val in [
    ("messages", []), ("retriever", None), ("vectorstore_ready", False),
    ("uploaded_file_names", []), ("quiz_data", None),
    ("active_tab", "chat"), ("workspace_open", False)
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Nav ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="ws-nav a1">
    <div class="ws-logo">
        <span class="ws-dot"></span>
        Study<span style="color:{ACCENT}">Mind</span> AI
    </div>
    <div style="font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;color:{TEXT2};">Workspace</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-section a2">
    <div class="hero-bg">AI</div>
    <div class="hero-eye">● StudyMind AI · RAG Pipeline</div>
    <div class="hero-title">Your personal <em>study assistant.</em></div>
    <div class="hero-sub">
        Upload your PDFs, ask anything in plain English, and auto-generate quizzes —
        all powered by a local RAG pipeline. No hallucinations. Just your material.
    </div>
</div>
""", unsafe_allow_html=True)

# Stats
st.markdown(f"""
<div class="stats-row a3">
    <div class="stat-item"><div class="stat-num">384</div><div class="stat-lbl">Embedding dims</div></div>
    <div class="stat-item"><div class="stat-num">512</div><div class="stat-lbl">Tokens/chunk</div></div>
    <div class="stat-item"><div class="stat-num">∞</div><div class="stat-lbl">PDFs supported</div></div>
    <div class="stat-item"><div class="stat-num">Top 5</div><div class="stat-lbl">Retrieved chunks</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ── Upload Section ─────────────────────────────────────────────────────────────
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

# Feature strip
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

# Process button
if uploaded_files:
    new_names = sorted([f.name for f in uploaded_files])
    already = sorted(st.session_state.uploaded_file_names)
    if new_names != already:
        if st.button("⚡ Process PDFs"):
            with st.spinner("Processing your PDFs..."):
                os.makedirs("data", exist_ok=True)

                saved_paths = []

                for f in uploaded_files:
                    file_path = os.path.join("data", f.name)

                    with open(file_path, "wb") as out:
                        out.write(f.read())

                    saved_paths.append(file_path)

                docs = load_docs(file_paths=saved_paths)

                chunks = split_docs(docs)

                embedding_manager = EmbeddingManager()
                embeddings = embedding_manager.generate_embedding(
                    [d.page_content for d in chunks]
                )

                vector_store = VectorStoreManager()
                vector_store.add_documents(chunks, embeddings)

                st.session_state.retriever = RAGRetriever(
                    embedding_manager,
                    vector_store
                )

                st.session_state.vectorstore_ready = True
                st.session_state.messages = []
                st.session_state.quiz_data = None
                st.session_state.uploaded_file_names = [f.name for f in uploaded_files]
                st.session_state.workspace_open = True

            st.success(f"✅ {len(uploaded_files)} PDF(s) ready!")
    else:
        st.success(f"✅ {len(uploaded_files)} PDF(s) already loaded!")
        if st.button("🚀 Open Workspace"):
            st.session_state.workspace_open = True
            st.rerun()

# File badges
if st.session_state.vectorstore_ready:
    badges = "".join([f'<span class="fbadge">📄 {n}</span>' for n in st.session_state.uploaded_file_names])
    st.markdown(f'<div style="margin-top:14px;">{badges}</div>', unsafe_allow_html=True)

# ── Workspace ──────────────────────────────────────────────────────────────────
if not st.session_state.workspace_open or not st.session_state.vectorstore_ready:
    st.stop()

st.markdown(f'<div style="border-top:1px solid {BORDER};margin:48px 0 0;"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="workspace-section">
    <div class="s-eye">● Step 02</div>
    <div class="s-title">Chat or <em>test yourself.</em></div>
</div>
""", unsafe_allow_html=True)

# Tab switcher buttons
col_c, col_q, col_clear, col_sp = st.columns([1, 1, 1, 5])
with col_c:
    if st.button("💬 Chat", key="btn_chat"):
        st.session_state.active_tab = "chat"
        st.rerun()
with col_q:
    if st.button("🧩 Quiz", key="btn_quiz"):
        st.session_state.active_tab = "quiz"
        st.rerun()
with col_clear:
    if st.button("🗑️ Clear", key="btn_clear"):
        st.session_state.messages = []
        st.session_state.quiz_data = None
        st.rerun()

st.markdown(f'<div style="border-bottom:1px solid {BORDER};margin:8px 0 28px;"></div>', unsafe_allow_html=True)

# ── Chat ───────────────────────────────────────────────────────────────────────
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
                    answer = generate_output(prompt, st.session_state.retriever)
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Quiz ───────────────────────────────────────────────────────────────────────
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

        if st.button("⚡ Generate Quiz"):
            with st.spinner("Generating..."):
                from quiz import generate_quiz
                st.session_state.quiz_data = generate_quiz(
                    retriever=st.session_state.retriever,
                    topic=topic,
                    num_questions=num_q
                )

    if st.session_state.quiz_data:
        st.markdown(f'<div style="border-top:1px solid {BORDER};margin:24px 0;"></div>', unsafe_allow_html=True)
        answers = {}
        for i, q in enumerate(st.session_state.quiz_data):
            st.markdown('<div class="card" style="margin:10px 0;">', unsafe_allow_html=True)
            st.markdown(f"**Q{i+1}. {q['question']}**")
            answers[i] = st.radio("", q["options"], key=f"q{i}", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✅ Submit Quiz"):
            score = sum(1 for i, q in enumerate(st.session_state.quiz_data) if answers[i] == q["answer"])
            total = len(st.session_state.quiz_data)
            msg = "Perfect score! 🔥" if score == total else "Good job! Keep revising 💪" if score >= total//2 else "Need more practice!"
            st.markdown(f"""
            <div class="score-card">
                <div class="score-num">{score} / {total}</div>
                <div class="score-msg">{msg}</div>
            </div>
            """, unsafe_allow_html=True)
            if score == total:
                st.balloons()