"""
PDF Chat Assistant — Streamlit RAG Application
- In-memory PDF text extraction (pypdf.PdfReader)
- In-memory Chroma vectorstore (langchain-chroma)
- Fast offline HuggingFace embeddings (HF_HUB_OFFLINE=1)
"""

import os
import time

# Set offline environment flags BEFORE importing HuggingFace / Transformers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------------------
# Page config — MUST be the first Streamlit command executed
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# Header — Renders instantly
# --------------------------------------------------------------------------------------
st.title("📄 PDF Chat Assistant")
st.caption("Upload a PDF document to ask grounded questions with page number citations.")
st.divider()

# --------------------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None

# --------------------------------------------------------------------------------------
# Lazy Loaders for Heavy ML Packages
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing local embedding engine...")
def get_embeddings():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from langchain_huggingface import HuggingFaceEmbeddings
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm(model_name: str):
    from langchain_mistralai import ChatMistralAI
    api_key = os.getenv("MISTRAL_API_KEY") or st.session_state.get("mistral_api_key")
    if not api_key:
        return None
    return ChatMistralAI(model=model_name, api_key=api_key)

# --------------------------------------------------------------------------------------
# Sidebar — API Configuration & Model Selector & File Uploader
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("🔑 Mistral API Key")
    env_api_key = os.getenv("MISTRAL_API_KEY", "")
    active_key = st.session_state.get("mistral_api_key", env_api_key)

    if active_key:
        st.success("🟢 API Key Active")
        with st.expander("Update API Key"):
            new_key = st.text_input("New API Key", type="password", key="new_key_field")
            if st.button("Save Key", use_container_width=True):
                st.session_state["mistral_api_key"] = new_key
                os.environ["MISTRAL_API_KEY"] = new_key
                st.rerun()
    else:
        st.info("💡 Enter your Mistral API key to run queries.")
        user_key = st.text_input(
            "Enter Key",
            type="password",
            placeholder="Enter Mistral API key...",
            help="Get your key at https://console.mistral.ai",
        )
        if user_key:
            st.session_state["mistral_api_key"] = user_key
            os.environ["MISTRAL_API_KEY"] = user_key
            st.rerun()

    st.subheader("🤖 Model Selection")
    selected_model = st.selectbox(
        "Choose Mistral Model",
        options=["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest", "open-mistral-7b"],
        index=0,
    )

    st.divider()
    st.header("📎 Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    process_clicked = st.button("⚡ Process Document", use_container_width=True, type="primary")

    st.divider()
    if st.session_state.retriever is not None:
        st.success(f" Ready — **{st.session_state.doc_name}**")
        if st.session_state.doc_stats:
            c1, c2 = st.columns(2)
            c1.metric("Pages", st.session_state.doc_stats["total_pages"])
            c2.metric("Chunks", st.session_state.doc_stats["total_chunks"])

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col_b:
            if st.button("🗑️ Unload PDF", use_container_width=True):
                st.session_state.retriever = None
                st.session_state.doc_name = None
                st.session_state.doc_stats = None
                st.session_state.messages = []
                st.rerun()
    else:
        st.warning("⏳ Waiting for a PDF document...")

# --------------------------------------------------------------------------------------
# In-Memory Document Processing Pipeline
# --------------------------------------------------------------------------------------
if process_clicked:
    if uploaded_file is None:
        st.warning("⚠️ Please select a PDF file first before clicking Process Document.")
    else:
        progress_bar = st.progress(0, text="Starting in-memory PDF extraction...")
        try:
            # Step 1: Read PDF pages directly from memory buffer using pypdf
            progress_bar.progress(25, text="Reading PDF pages directly from memory...")
            import pypdf
            from langchain_core.documents import Document

            pdf_reader = pypdf.PdfReader(uploaded_file)
            docs = []
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    docs.append(Document(page_content=text, metadata={"page": page_num}))

            if not docs:
                progress_bar.empty()
                st.warning("⚠️ No selectable text was found in this PDF. Please ensure your PDF contains selectable text (not scanned images).")
            else:
                # Step 2: Split text into chunks
                progress_bar.progress(55, text=f"Extracted {len(docs)} pages. Splitting text into semantic chunks...")
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                texts = text_splitter.split_documents(docs)

                # Step 3: Embed & Index in In-Memory Chroma
                progress_bar.progress(85, text=f"Indexing {len(texts)} chunks into in-memory Chroma vectorstore...")
                embeddings = get_embeddings()
                from langchain_chroma import Chroma

                vector_store = Chroma.from_documents(
                    documents=texts,
                    embedding=embeddings,
                )

                retriever = vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
                )

                st.session_state.retriever = retriever
                st.session_state.doc_name = uploaded_file.name
                st.session_state.doc_stats = {
                    "total_pages": len(docs),
                    "total_chunks": len(texts),
                }
                st.session_state.messages = []

                progress_bar.progress(100, text="Complete!")
                time.sleep(0.3)
                progress_bar.empty()
                st.success(f"🎉 **{uploaded_file.name}** processed successfully! ({len(docs)} pages, {len(texts)} chunks indexed)")
        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ Error processing PDF: {str(e)}")

# --------------------------------------------------------------------------------------
# Main Area Banner when no document loaded
# --------------------------------------------------------------------------------------
if st.session_state.retriever is None:
    st.info("👈 **Get Started**: Upload a PDF file in the sidebar on the left and click **⚡ Process Document**.")
    st.markdown(
        """
        ### App Features:
        - 📄 **In-Memory PDF Text Extraction** via `pypdf`.
        - ⚡ **In-Memory Chroma Vectorstore** for fast similarity search.
        - 🤖 **Mistral AI Integration** for precise Q&A with page number citations.
        - 📚 **Source Inspector** to view exact document passages used for each answer.
        """
    )

# --------------------------------------------------------------------------------------
# Chat History Display with Expandable Sources
# --------------------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 View Retrieved Source Passages & Pages"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {idx} (Page {src['page']}):**")
                    st.caption(src["text"])

# --------------------------------------------------------------------------------------
# Chat Input & Response Pipeline
# --------------------------------------------------------------------------------------
placeholder = (
    "Ask a question about your document..."
    if st.session_state.retriever is not None
    else "Upload and process a PDF first..."
)
user_input = st.chat_input(placeholder, disabled=st.session_state.retriever is None)

if user_input:
    current_llm = get_llm(selected_model)
    if not current_llm:
        st.warning("⚠️ Please provide a valid Mistral API key in the sidebar before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document and generating answer..."):
                try:
                    docs = st.session_state.retriever.invoke(user_input)

                    context_blocks = []
                    sources_list = []
                    for doc in docs:
                        page_num = doc.metadata.get("page", 0) + 1
                        context_blocks.append(f"[Page {page_num}]:\n{doc.page_content}")
                        sources_list.append({"page": page_num, "text": doc.page_content})
                    context = "\n\n".join(context_blocks)

                    from langchain_core.prompts import ChatPromptTemplate
                    prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "You are a helpful assistant. Use the following pieces of context to answer "
                                "the question at the end. If you don't know the answer, just say that you "
                                "don't know, don't try to make up an answer. Mention page numbers "
                                "where concepts appear when relevant.",
                            ),
                            ("human", "context:\n{context}\n\nquestion: {question}"),
                        ]
                    )

                    final_prompt = prompt.invoke({"context": context, "question": user_input})
                    response = current_llm.invoke(final_prompt)
                    full_answer = str(response.content)

                    # Reveal output with typing animation
                    placeholder_area = st.empty()
                    rendered = ""
                    for ch in full_answer:
                        rendered += ch
                        placeholder_area.markdown(rendered + "▌")
                        time.sleep(0.005)
                    placeholder_area.markdown(rendered)

                    # Show source chunks accordion
                    with st.expander("📚 View Retrieved Source Passages & Pages"):
                        for idx, src in enumerate(sources_list, 1):
                            st.markdown(f"**Source {idx} (Page {src['page']}):**")
                            st.caption(src["text"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer,
                        "sources": sources_list,
                    })
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")