import os
import tempfile
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # reads GROQ_API_KEY from a local .env file

APP_NAME = "StudyBuddy"
NOT_FOUND = "Sorry, answer not found"
MODEL_NAME = "openai/gpt-oss-20b"
HISTORY_TURNS = 3  # past question/answer pairs sent along for follow-up questions

SYSTEM_PROMPT = f"""You are {APP_NAME}, a friendly study assistant.
Rules:
- Answer using ONLY the PDF context provided. Do not use outside knowledge or guess.
- If the answer is not clearly in the context, reply exactly: "{NOT_FOUND}".
- Brief small talk (hi, hello, thanks, who are you) is fine: reply in one short, friendly line and invite a question about the PDF.
- If the user keeps chatting about unrelated topics, politely steer them back to the document.
"""

st.set_page_config(page_title=APP_NAME, page_icon="📚")
st.title(f"📚 {APP_NAME}")
st.caption("Upload a PDF and ask questions. I only answer from what's in your document.")


# ---------- Helpers ----------
def get_api_key():
    """Streamlit Cloud secret first, then local .env / environment variable."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_llm(api_key: str):
    return ChatGroq(model=MODEL_NAME, api_key=api_key, temperature=0)


def build_retriever(pdf_bytes: bytes):
    """Read the PDF, split it, embed it and return (retriever, n_pages, n_chunks)."""
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(pdf_bytes)
            path = f.name
        pages = PyPDFLoader(path).load()
    finally:
        if path and os.path.exists(path):
            os.remove(path)

    pages = [p for p in pages if p.page_content.strip()]
    if not pages:
        raise ValueError(
            "No readable text found. This looks like a scanned PDF, which needs OCR first."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    # In-memory Chroma with a unique collection name, so different
    # users/uploads never share or mix their documents.
    db = Chroma.from_documents(
        chunks, get_embeddings(), collection_name=f"pdf_{uuid4().hex}"
    )
    retriever = db.as_retriever(
        search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )
    return retriever, len(pages), len(chunks)


def format_history(messages):
    recent = messages[-HISTORY_TURNS * 2:]
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)


def ask(question: str, api_key: str):
    docs = st.session_state.retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    history = format_history(st.session_state.messages) or "(none)"
    user_msg = (
        f"Chat so far:\n{history}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    response = get_llm(api_key).invoke([("system", SYSTEM_PROMPT), ("human", user_msg)])
    pages = sorted({d.metadata.get("page", 0) + 1 for d in docs})
    return response.content, pages


def show_message(role: str, content: str, sources=None):
    with st.chat_message(role):
        st.write(content)
        if sources:
            st.caption("📄 Pages: " + ", ".join(str(p) for p in sources))


# ---------- API key check ----------
api_key = get_api_key()
if not api_key:
    st.error(
        "GROQ_API_KEY not found. Locally, add it to a `.env` file "
        "(`GROQ_API_KEY=your_key`). On Streamlit Cloud, add it under Settings → Secrets."
    )
    st.stop()

# ---------- Sidebar ----------
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("Your document")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    if "pdf_name" in st.session_state:
        st.success(
            f"{st.session_state.pdf_name}\n\n"
            f"{st.session_state.n_pages} pages · {st.session_state.n_chunks} chunks"
        )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Load / rebuild when a new PDF is chosen ----------
if uploaded is None:
    for key in ("retriever", "pdf_name", "n_pages", "n_chunks"):
        st.session_state.pop(key, None)
elif st.session_state.get("pdf_name") != uploaded.name:
    try:
        with st.spinner(f"{APP_NAME} is reading your PDF..."):
            retriever, n_pages, n_chunks = build_retriever(uploaded.getvalue())
        st.session_state.retriever = retriever
        st.session_state.pdf_name = uploaded.name
        st.session_state.n_pages = n_pages
        st.session_state.n_chunks = n_chunks
        st.session_state.messages = []
        st.rerun()
    except Exception as e:
        st.error(f"Could not read this PDF: {e}")
        st.stop()

# ---------- Chat ----------
for m in st.session_state.messages:
    show_message(m["role"], m["content"], m.get("sources"))

ready = "retriever" in st.session_state
if not ready:
    st.info("👈 Upload a PDF in the sidebar to get started.")

question = st.chat_input(
    f"Ask {APP_NAME} about {st.session_state.get('pdf_name', 'your PDF')}",
    disabled=not ready,
)

if question:
    show_message("user", question)
    try:
        with st.spinner("Thinking..."):
            answer, pages = ask(question, api_key)
    except Exception as e:
        st.error(f"Something went wrong while answering: {e}")
        st.stop()

    sources = None if answer.strip().rstrip(".") == NOT_FOUND else pages
    show_message("assistant", answer, sources)
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "sources": sources},
    ]
