import json
import os
from pathlib import Path
import uuid

import requests
import streamlit as st


st.set_page_config(
    page_title="Vivasaya Thozan AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = (5, 180)
AGRI_IMAGE_PATH = Path(__file__).with_name("agri_img.png")
if not AGRI_IMAGE_PATH.is_file():
    raise FileNotFoundError(f"Required UI background image not found: {AGRI_IMAGE_PATH}")
AGRI_STATIC_DIR = Path(__file__).with_name("static")
AGRI_STATIC_IMAGE_PATH = AGRI_STATIC_DIR / "agri_img.png"
AGRI_HERO_IMAGE_PATH = AGRI_STATIC_DIR / "agri_hero.png"
if not AGRI_STATIC_IMAGE_PATH.is_file() or not AGRI_HERO_IMAGE_PATH.is_file():
    raise FileNotFoundError(
        f"Required Streamlit image assets were not found in {AGRI_STATIC_DIR}"
    )
SAMPLE_QUESTIONS = [
    ("🌻", "Budget", "How can I plan a crop budget for 10 acres?"),
    ("📄", "Schemes", "Which agriculture schemes are available in Tamil Nadu?"),
    ("📈", "Policy", "What are the key initiatives in the agriculture budget?"),
    ("🌱", "Future plans", "Which crops suit low rainfall and loam soil?"),
    ("🌾", "Farm advice", "How can I improve organic carbon in my farm soil?"),
    ("🏛️", "Leadership", "Who is the Chief Minister of Tamil Nadu?"),
]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    :root {
        --forest: #103b2d;
        --forest-deep: #09271e;
        --leaf: #28734f;
        --leaf-light: #e9f3e8;
        --gold: #d9b969;
        --cream: #f8f7ef;
        --ink: #20352b;
        --muted: #64736a;
        --line: #e1e8de;
        --agri-image: url("/app/static/agri_img.png");
    }
    .stApp,
    [data-testid="stAppViewContainer"] {
        background-image:
            linear-gradient(rgba(248,247,239,.76), rgba(238,244,234,.8)),
            var(--agri-image);
        background-attachment: fixed;
        background-position: center, center;
        background-size: cover, cover;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--forest-deep), var(--forest) 70%, #174c38);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] * { color: #f0f5eb; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 10px;
        padding: .3rem .5rem;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        border: 1px dashed rgba(240,245,235,.38);
        border-radius: 12px;
        background: rgba(255,255,255,.055);
        padding: .55rem;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,.07);
        color: #f0f5eb !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background: #f4f5e9;
        color: var(--forest) !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
        color: var(--forest) !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #d7e3d5 !important;
    }
    .brand-lockup { padding: .3rem 0 1.2rem; }
    .brand-kicker {
        color: #d9e7d7;
        font-size: .69rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .brand-name {
        color: #f7f3df;
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        font-weight: 700;
        line-height: 1.2;
        margin-top: .4rem;
    }
    .brand-subtitle { color: #bed1c0; font-size: .76rem; margin-top: .35rem; }
    .section-label {
        color: #c6d5c5;
        font-size: .67rem;
        font-weight: 700;
        letter-spacing: .14em;
        margin: 1rem 0 .45rem;
        text-transform: uppercase;
    }
    .metric-card {
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 12px;
        margin: .45rem 0;
        padding: .7rem .8rem;
    }
    .metric-label { color: #c2d3c3; font-size: .72rem; }
    .metric-value { color: #fff; font-size: 1.17rem; font-weight: 700; margin-top: .12rem; }
    .hero {
        background: linear-gradient(112deg, #0b2e23, #14513a 64%, #256b49);
        border: 1px solid rgba(255,255,255,.1);
        border-radius: 22px;
        box-shadow: 0 16px 36px rgba(16,59,45,.16);
        margin: .15rem 0 1.2rem;
        overflow: hidden;
        padding: clamp(1.35rem, 3vw, 2.3rem);
        position: relative;
    }
    .hero-layout {
        align-items: center;
        display: grid;
        gap: clamp(1rem, 3vw, 2.5rem);
        grid-template-columns: minmax(0, 1.35fr) minmax(220px, .8fr);
    }
    .hero-copy { position: relative; z-index: 1; }
    .hero-visual {
        background-image:
            linear-gradient(90deg, rgba(11,46,35,.05), rgba(11,46,35,.18)),
            url("/app/static/agri_hero.png");
        background-position: center;
        background-repeat: no-repeat;
        background-size: cover;
        border: 1px solid rgba(255,255,255,.25);
        border-radius: 16px;
        box-shadow: 0 12px 28px rgba(0,0,0,.18);
        height: clamp(170px, 17vw, 225px);
        min-width: 0;
    }
    .hero-eyebrow {
        color: #d7c782;
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .2em;
        text-transform: uppercase;
    }
    .hero-title {
        color: #f7f1d9;
        font-family: 'Playfair Display', serif;
        font-size: clamp(2rem, 4.2vw, 3.25rem);
        font-weight: 700;
        line-height: 1.1;
        margin: .35rem 0;
    }
    .hero-subtitle { color: #d6e4d3; font-size: .96rem; }
    .hero-pill {
        background: rgba(255,255,255,.1);
        border: 1px solid rgba(255,255,255,.18);
        border-radius: 999px;
        color: #f2f0db;
        display: inline-block;
        font-size: .76rem;
        margin-top: 1rem;
        padding: .38rem .75rem;
    }
    .panel {
        background: rgba(255,255,255,.94);
        border: 1px solid var(--line);
        border-radius: 18px;
        box-shadow: 0 8px 26px rgba(28,61,42,.06);
        padding: 1.1rem 1.2rem;
    }
    .panel-heading {
        color: var(--forest);
        font-size: 1.08rem;
        font-weight: 700;
        margin-bottom: .2rem;
    }
    .panel-caption { color: var(--muted); font-size: .8rem; margin-bottom: .85rem; }
    .question-card {
        background: linear-gradient(145deg, #fbfcf7, #f0f5eb);
        border: 1px solid #e1e9dc;
        border-radius: 13px;
        color: var(--ink);
        min-height: 78px;
        padding: .72rem .8rem;
    }
    .question-title { color: var(--forest); font-size: .84rem; font-weight: 700; }
    .question-text { color: #526258; font-size: .75rem; line-height: 1.35; margin-top: .3rem; }
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 2.65rem;
        transition: all .18s ease;
    }
    .stButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #28734f, #1c5d3d);
        border: 1px solid #1c5d3d;
        color: white;
    }
    .stButton > button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background: #164b34;
        border-color: #164b34;
        box-shadow: 0 5px 14px rgba(40,115,79,.2);
        color: white;
    }
    [data-testid="stChatMessage"] {
        background: rgba(248,250,245,.84);
        border: 1px solid #e8ede5;
        border-radius: 14px;
        margin-bottom: .55rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #edf5e8;
    }
    [data-testid="stTextArea"] textarea {
        background: #fff;
        border: 1px solid #dce5d8;
        border-radius: 11px;
        height: 2.8rem !important;
        min-height: 2.8rem;
        max-height: 15rem;
        overflow-y: auto;
        resize: vertical;
        transition: height .2s ease, box-shadow .2s ease;
    }
    [data-testid="stTextArea"]:hover textarea,
    [data-testid="stTextArea"]:focus-within textarea {
        height: 9rem !important;
        box-shadow: 0 0 0 3px rgba(40,115,79,.12);
    }
    .wisdom {
        background: linear-gradient(110deg, #103b2d, #1d6344);
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 17px;
        color: #f8f3df;
        margin-top: 1rem;
        padding: 1rem 1.2rem;
    }
    .wisdom-title { color: #e3d08b; font-weight: 700; margin-bottom: .3rem; }
    .wisdom-copy { color: #e7eddf; font-size: .82rem; }
    .footer-note { color: #728078; font-size: .72rem; margin-top: 1rem; text-align: center; }
    @media (max-width: 800px) {
        .panel { padding: .85rem; }
    }
    @media (max-width: 640px) {
        .hero-layout { grid-template-columns: 1fr; }
        .hero-visual { height: 150px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_stats():
    response = requests.get(f"{API_BASE_URL}/api/stats", timeout=5)
    response.raise_for_status()
    return response.json()


def upload_pdf(uploaded_file):
    response = requests.post(
        f"{API_BASE_URL}/api/upload",
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            )
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def process_uploaded_pdf(filename: str):
    response = requests.post(
        f"{API_BASE_URL}/api/process-upload",
        data={"filename": filename},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def submit_feedback(response_id: str, rating: str):
    response = requests.post(
        f"{API_BASE_URL}/api/feedback",
        data={
            "response_id": response_id,
            "rating": rating,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def stream_chat_response(question: str):
    with requests.post(
        f"{API_BASE_URL}/api/chat/stream",
        data={
            "question": question,
            "session_id": st.session_state.session_id,
        },
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                return
            payload = json.loads(data)
            if "error" in payload:
                raise RuntimeError(payload["error"])
            token = payload.get("token")
            if not isinstance(token, str):
                raise ValueError("The chat API returned an invalid response event.")
            if token:
                yield token


def ask_question(question: str):
    st.session_state.pending_question = question


if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Vanakkam! I’m **Vivasaya Thozan AI**, your agricultural companion. "
                "Ask about farming, Tamil Nadu schemes, or the documents in the knowledge base."
            ),
            "feedback_enabled": False,
        }
    ]
if "pending_upload_name" not in st.session_state:
    st.session_state.pending_upload_name = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-kicker">🌾 Vivasaya Thozan AI</div>
            <div class="brand-name">Cultivating tomorrow,<br>together.</div>
            <div class="brand-subtitle">Your intelligent agricultural companion</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Workspace</div>', unsafe_allow_html=True)
    active_page = st.radio(
        "Navigate",
        [
            "🏠  Home",
            "📊  Dashboard",
            "🌱  Crop Advisor",
            "💧  Soil & Weather",
            "📈  Market Insights",
            "📁  Farm Records",
            "👥  Community",
            "🔔  Notifications",
        ],
        label_visibility="collapsed",
        key="navigation",
    )
    st.markdown('<div class="section-label">Knowledge base</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a PDF from your device",
        type=["pdf"],
        key="pdf_uploader",
        help="PDFs are uploaded to the app server; adding them to search is a separate step.",
    )
    if st.button(
        "⬆️  Upload PDF",
        use_container_width=True,
        disabled=uploaded_file is None,
        key="upload_pdf_button",
    ):
        try:
            result = upload_pdf(uploaded_file)
            st.session_state.pending_upload_name = result["filename"]
            st.success(f"Uploaded {result['filename']}.")
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Upload failed: {detail}")
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")

    pending_upload = st.session_state.pending_upload_name
    if pending_upload:
        st.caption(f"Ready to process: **{pending_upload}**")
    if st.button(
        "⚡  Create chunks & add to vector store",
        type="primary",
        use_container_width=True,
        disabled=not pending_upload,
        key="process_pdf_button",
    ):
        with st.spinner("Extracting PDF content and creating searchable chunks…"):
            try:
                result = process_uploaded_pdf(pending_upload)
                st.session_state.pending_upload_name = None
                st.success(result.get("message", "PDF processed successfully."))
                st.rerun()
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Chunk generation failed: {detail}")
            except requests.RequestException as exc:
                st.error(f"Chunk generation failed: {exc}")

    st.markdown('<div class="section-label">Knowledge base status</div>', unsafe_allow_html=True)
    try:
        stats = fetch_stats()
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Searchable chunks</div>
                <div class="metric-value">{stats.get("vector_db_chunks", 0):,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Indexed documents</div>
                <div class="metric-value">{stats.get("knowledge_base_count", 0):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except requests.RequestException:
        st.warning("Backend unavailable. Start the API to see live index status.")

    st.markdown(
        '<div class="footer-note">Grounded answers from your indexed farm documents</div>',
        unsafe_allow_html=True,
    )

if active_page != "🏠  Home":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Vivasaya Thozan AI · Workspace</div>
            <div class="hero-title">Your farm, at a glance.</div>
            <div class="hero-subtitle">This workspace is coming soon. Your document assistant is ready on Home.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Choose **Home** in the sidebar to chat with the agricultural document assistant.")
    st.stop()

st.markdown(
    """
    <div class="hero">
        <div class="hero-layout">
            <div class="hero-copy">
                <div class="hero-eyebrow">🌾 Vivasaya Thozan AI</div>
                <div class="hero-title">Grow with confidence.<br>Thrive with every season.</div>
                <div class="hero-subtitle">Trusted agricultural knowledge to help you make informed decisions, strengthen your farm, and plan what comes next.</div>
                <div class="hero-pill">Tamil Nadu agriculture · Policies · Schemes · Farm advice</div>
            </div>
            <div class="hero-visual" role="img" aria-label="Farmers working in green agricultural fields"></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

chat_column, questions_column = st.columns([1.65, 1], gap="large")
with chat_column:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-heading">🌿 Chat with Vivasaya Thozan AI</div>
            <div class="panel-caption">Answers are grounded in your indexed documents and include source references.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chat_container = st.container(height=430, border=True)
    with chat_container:
        for message in st.session_state.messages:
            avatar = "🧑‍🌾" if message["role"] == "user" else "🌾"
            if message["role"] == "assistant":
                message.setdefault("response_id", uuid.uuid4().hex)
                message.setdefault("feedback_enabled", True)
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message["feedback_enabled"]:
                    feedback_columns = st.columns([0.14, 0.14, 0.72])
                    with feedback_columns[0]:
                        if st.button(
                            "👍",
                            key=f"like_{message['response_id']}",
                            help="This answer was helpful",
                            type=(
                                "primary"
                                if message.get("feedback") == "like"
                                else "secondary"
                            ),
                        ):
                            try:
                                submit_feedback(message["response_id"], "like")
                                message["feedback"] = "like"
                            except requests.RequestException as exc:
                                st.error(f"Could not save feedback: {exc}")
                    with feedback_columns[1]:
                        if st.button(
                            "👎",
                            key=f"dislike_{message['response_id']}",
                            help="This answer was not helpful",
                            type=(
                                "primary"
                                if message.get("feedback") == "dislike"
                                else "secondary"
                            ),
                        ):
                            try:
                                submit_feedback(message["response_id"], "dislike")
                                message["feedback"] = "dislike"
                            except requests.RequestException as exc:
                                st.error(f"Could not save feedback: {exc}")
                    with feedback_columns[2]:
                        if message.get("feedback"):
                            st.caption("Thanks for your feedback.")

    with st.form("chat_form", clear_on_submit=True):
        input_column, button_column = st.columns([4, 1.25], vertical_alignment="bottom")
        with input_column:
            user_question = st.text_area(
                "Your question",
                placeholder="Ask anything about agriculture…",
                height=68,
                label_visibility="collapsed",
                key="chat_question",
            )
        with button_column:
            submitted = st.form_submit_button("Get answer →", use_container_width=True)

with questions_column:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-heading">✨ Questions to get you started</div>
            <div class="panel-caption">Choose a topic, or ask your own question.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for index in range(0, len(SAMPLE_QUESTIONS), 2):
        question_columns = st.columns(2, gap="small")
        for offset, question_data in enumerate(
            SAMPLE_QUESTIONS[index:index + 2]
        ):
            icon, title, question = question_data
            with question_columns[offset]:
                st.markdown(
                    f"""
                    <div class="question-card">
                        <div class="question-title">{icon} &nbsp; {title}</div>
                        <div class="question-text">{question}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    "Ask this →",
                    key=f"sample_question_{index + offset}",
                    use_container_width=True,
                    on_click=ask_question,
                    args=(question,),
                )

question_to_send = None
if submitted and user_question.strip():
    question_to_send = user_question.strip()
elif st.session_state.pending_question:
    question_to_send = st.session_state.pending_question
    st.session_state.pending_question = None

if question_to_send:
    st.session_state.messages.append({"role": "user", "content": question_to_send})
    with chat_container:
        with st.chat_message("assistant", avatar="🌾"):
            response_placeholder = st.empty()
            answer = ""
            try:
                for token in stream_chat_response(question_to_send):
                    answer += token
                    response_placeholder.markdown(f"{answer}▌")
                if not answer:
                    raise RuntimeError("The chat API returned an empty answer.")
                response_placeholder.markdown(answer)
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                answer = f"Sorry, I couldn’t generate a response. {exc}"
                response_placeholder.error(answer)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "response_id": uuid.uuid4().hex,
            "feedback_enabled": not answer.startswith(
                "Sorry, I couldn’t generate a response."
            ),
        }
    )
    st.rerun()

st.markdown(
    """
    <div class="wisdom">
        <div class="wisdom-title">🌱 Tamil agricultural wisdom</div>
        <div class="wisdom-copy">“மண் வளம் காப்போம், நாளை சிறப்போம்.” &nbsp; · &nbsp;
        Protect the fertility of the soil, and we shall prosper tomorrow.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
