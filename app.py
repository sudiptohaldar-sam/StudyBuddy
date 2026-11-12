import tempfile

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

APP_NAME = "StudyBuddy"
NOT_FOUND = "Sorry, answer not found"

st.set_page_config(page_title=APP_NAME, page_icon="📚")
st.title(f"📚 {APP_NAME}")
st.caption("Upload a PDF and ask questions. I only answer from what's in your document.")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_llm():
    try:
        key = st.secrets["GROQ_API_KEY"]   # Streamlit Cloud
    except Exception:
        key = os.getenv("GROQ_API_KEY")    # local .env file
    return ChatGroq(model="openai/gpt-oss-20b", api_key=key)


def build_retriever(pdf_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        path = f.name

    docs = PyPDFLoader(path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # In-memory Chroma: no folder on disk, so it works on Streamlit Cloud
    db = Chroma.from_documents(chunks, get_embeddings())
    return db.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})


def ask(question: str) -> str:
    docs = st.session_state.retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = (
        f"You are {APP_NAME}, a study assistant. Answer ONLY using the context below. "
        f"If the answer is not in the context, reply exactly: {NOT_FOUND}\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    return get_llm().invoke(prompt).content


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Your document")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Load / rebuild when a new PDF is chosen ----------
if uploaded and st.session_state.get("pdf_name") != uploaded.name:
    with st.spinner(f"{APP_NAME} is reading your PDF..."):
        st.session_state.retriever = build_retriever(uploaded.getvalue())
        st.session_state.pdf_name = uploaded.name
        st.session_state.messages = []

# ---------- Chat ----------
st.session_state.setdefault("messages", [])

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

if "retriever" not in st.session_state:
    st.info("👈 Upload a PDF in the sidebar to get started.")
elif question := st.chat_input(f"Ask {APP_NAME} about {st.session_state.pdf_name}"):
    st.chat_message("user").write(question)
    with st.spinner("Thinking..."):
        answer = ask(question)
    st.chat_message("assistant").write(answer)
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
