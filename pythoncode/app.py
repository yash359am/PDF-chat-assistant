"""
PDF Chat Assistant — Streamlit UI wrapper around the original LangChain + Mistral RAG pipeline.
The retrieval / prompting / LLM logic is unchanged from the original script:
    - HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
    - Chroma vector store
    - MMR retriever (k=4, fetch_k=10, lambda_mult=0.5)
    - ChatMistralAI("mistral-small-2603")
    - Same system/human prompt template
Only addition: a UI + the ability to upload a PDF, which replaces the commented-out
PyPDFLoader/text-splitter block in the original script with an interactive version.
"""

import os
import shutil
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

load_dotenv()

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# Styling — gradient header, animated chat bubbles, polished sidebar
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0%, #111827 45%, #0b1120 100%);
    }

    /* Animated gradient header */
    .app-header {
        padding: 28px 32px;
        border-radius: 18px;
        margin-bottom: 24px;
        background: linear-gradient(120deg, #6366f1, #8b5cf6, #06b6d4, #6366f1);
        background-size: 300% 300%;
        animation: gradientShift 10s ease infinite;
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.25);
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .app-header h1 {
        color: white;
        margin: 0;
        font-weight: 700;
        font-size: 1.9rem;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: rgba(255,255,255,0.85);
        margin: 6px 0 0 0;
        font-size: 0.95rem;
    }

    /* Status pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 10px;
    }
    .status-ready {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.35);
    }
    .status-waiting {
        background: rgba(250, 204, 21, 0.15);
        color: #facc15;
        border: 1px solid rgba(250, 204, 21, 0.35);
    }
    .dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: currentColor;
        animation: pulse 1.6s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.7); }
    }

    /* Chat bubbles fade-in */
    div[data-testid="stChatMessage"] {
        animation: fadeInUp 0.35s ease;
        border-radius: 14px;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0b1120 100%);
        border-right: 1px solid rgba(148,163,184,0.12);
    }

    .stButton>button {
        background: linear-gradient(120deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1rem;
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(99, 102, 241, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>📄 PDF Chat Assistant</h1>
        <p>Upload a PDF, then ask questions grounded in its content — powered by Mistral + LangChain.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = None

# --------------------------------------------------------------------------------------
# Cached embeddings + LLM (same models/params as the original script)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatMistralAI(model_name="mistral-small-2603")


embeddings = load_embeddings()
llm = load_llm()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use the following pieces of context to answer "
            "the question at the end. If you don't know the answer, just say that you "
            "don't know, don't try to make up an answer. and alos tell the page number "
            "which the concept is present  is the use ask",
        ),
        ("human", "context : {context} question:{question}"),
    ]
)

# --------------------------------------------------------------------------------------
# Sidebar — PDF upload & processing (this replaces the original commented-out block)
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📎 Upload document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    process_clicked = st.button("⚡ Process Document", use_container_width=True)

    if process_clicked and uploaded_file is not None:
        with st.spinner("Reading, chunking and embedding your PDF..."):
            # Save the upload to a temp file so PyPDFLoader can read it
            tmp_dir = tempfile.mkdtemp()
            pdf_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Same steps as the original commented-out block
            pdf_loader = PyPDFLoader(pdf_path)
            docs = pdf_loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
            texts = text_splitter.split_documents(docs)

            # Fresh persist directory per document so uploads don't mix together
            if st.session_state.persist_dir and os.path.exists(st.session_state.persist_dir):
                shutil.rmtree(st.session_state.persist_dir, ignore_errors=True)
            persist_directory = os.path.join(tmp_dir, "chroma_langchain_db")

            vector_store = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=persist_directory,
            )

            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
            )

            st.session_state.retriever = retriever
            st.session_state.doc_name = uploaded_file.name
            st.session_state.persist_dir = persist_directory
            st.session_state.messages = []

        st.success(f"Processed **{uploaded_file.name}** ({len(texts)} chunks indexed)")

    st.markdown("---")
    if st.session_state.retriever is not None:
        st.markdown(
            f'<div class="status-pill status-ready"><span class="dot"></span>'
            f'Ready — {st.session_state.doc_name}</div>',
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Clear document", use_container_width=True):
            if st.session_state.persist_dir and os.path.exists(st.session_state.persist_dir):
                shutil.rmtree(st.session_state.persist_dir, ignore_errors=True)
            st.session_state.retriever = None
            st.session_state.doc_name = None
            st.session_state.persist_dir = None
            st.session_state.messages = []
            st.rerun()
    else:
        st.markdown(
            '<div class="status-pill status-waiting"><span class="dot"></span>'
            "Waiting for a PDF</div>",
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------------------
# Chat history
# --------------------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------------------------------------------
# Chat input — same retrieval/prompt/LLM logic as the original while-loop
# --------------------------------------------------------------------------------------
placeholder = (
    "Ask a question about your document..."
    if st.session_state.retriever is not None
    else "Upload and process a PDF first..."
)
user_input = st.chat_input(placeholder, disabled=st.session_state.retriever is None)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = st.session_state.retriever.invoke(user_input)
            context = "".join([doc.page_content for doc in docs])
            final = prompt.invoke({"context": context, "question": user_input})
            response = llm.invoke(final)
            full_answer = response.content

        # Simple animated "typing" reveal
        placeholder_area = st.empty()
        rendered = ""
        for ch in full_answer:
            rendered += ch
            placeholder_area.markdown(rendered + "▌")
            time.sleep(0.005)
        placeholder_area.markdown(rendered)

    st.session_state.messages.append({"role": "assistant", "content": full_answer})