import os
import io
import hashlib
from typing import List, Tuple

import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5


# -----------------------------
# Cached models
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    # Streamlit Cloud stores secrets in st.secrets.
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        return None

    return Groq(api_key=api_key)


# -----------------------------
# Session-state initialization
# -----------------------------
def initialize_state():
    defaults = {
        "documents": [],
        "chunks": [],
        "metadata": [],
        "index": None,
        "messages": [],
        "processed_hashes": set(),
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


# -----------------------------
# Text extraction
# -----------------------------
def extract_text(uploaded_file) -> str:
    """Extract text from TXT/MD/PDF files."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    return uploaded_file.getvalue().decode("utf-8", errors="ignore")


# -----------------------------
# Chunking
# -----------------------------
def split_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


# -----------------------------
# Build / update FAISS index
# -----------------------------
def rebuild_index():
    if not st.session_state.chunks:
        st.session_state.index = None
        return

    model = load_embedding_model()

    embeddings = model.encode(
        st.session_state.chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    st.session_state.index = index


def add_document(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if file_hash in st.session_state.processed_hashes:
        return 0

    text = extract_text(uploaded_file)

    if not text.strip():
        return 0

    chunks = split_text(text)

    for i, chunk in enumerate(chunks):
        st.session_state.chunks.append(chunk)
        st.session_state.metadata.append(
            {
                "source": uploaded_file.name,
                "chunk": i + 1,
            }
        )

    st.session_state.documents.append(uploaded_file.name)
    st.session_state.processed_hashes.add(file_hash)

    rebuild_index()

    return len(chunks)


# -----------------------------
# Semantic search
# -----------------------------
def search_documents(query: str, top_k: int = TOP_K) -> List[Tuple[str, dict, float]]:
    if st.session_state.index is None:
        return []

    model = load_embedding_model()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, st.session_state.index.ntotal)

    scores, indices = st.session_state.index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append(
            (
                st.session_state.chunks[idx],
                st.session_state.metadata[idx],
                float(score),
            )
        )

    return results


# -----------------------------
# Groq answer generation
# -----------------------------
def generate_answer(question: str, retrieved_context):
    client = get_groq_client()

    if client is None:
        return (
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit Cloud Secrets before using the AI answer."
        )

    context_parts = []

    for text, metadata, score in retrieved_context:
        context_parts.append(
            f"Source: {metadata['source']} | "
            f"Chunk: {metadata['chunk']} | "
            f"Similarity: {score:.3f}\n"
            f"{text}"
        )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """You are a helpful Retrieval-Augmented Generation assistant.

Answer the user's question using the supplied document context.

Rules:
1. Use the retrieved context as the primary source of truth.
2. If the answer is not present in the context, clearly say that the uploaded
   documents do not contain enough information.
3. Do not invent facts, citations, or sources.
4. Give a clear, useful answer.
5. When possible, mention the source filename that supports the answer.
"""

    user_prompt = f"""DOCUMENT CONTEXT:

{context}

USER QUESTION:
{question}

Answer based on the document context."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    return response.choices[0].message.content


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Knowledge Base")

    st.caption(
        "Upload PDF, TXT, or Markdown files. "
        "The app creates embeddings and searches them with FAISS."
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("📥 Process Documents", use_container_width=True):
            total_chunks = 0

            with st.spinner("Reading, chunking and indexing documents..."):
                for uploaded_file in uploaded_files:
                    total_chunks += add_document(uploaded_file)

            st.success(
                f"Processed {len(uploaded_files)} file(s) "
                f"and added {total_chunks} chunk(s)."
            )

    st.divider()

    st.subheader("Knowledge Base")

    st.metric("Documents", len(st.session_state.documents))
    st.metric("Chunks", len(st.session_state.chunks))

    if st.session_state.documents:
        st.write("**Files:**")
        for document in st.session_state.documents:
            st.caption(f"• {document}")

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.metadata = []
        st.session_state.index = None
        st.session_state.messages = []
        st.session_state.processed_hashes = set()
        st.rerun()

    st.divider()

    st.caption(f"LLM: `{MODEL_NAME}`")
    st.caption(f"Embeddings: `{EMBEDDING_MODEL}`")
    st.caption("Vector DB: FAISS")


# -----------------------------
# Main UI
# -----------------------------
st.title("🤖 RAG AI Assistant")
st.write(
    "Upload your documents and ask questions. "
    "The app retrieves relevant information from FAISS and uses Groq "
    "to generate the final answer."
)

if not get_groq_client():
    st.warning(
        "⚠️ GROQ_API_KEY is not configured yet. "
        "The document indexing interface will still work, but AI answers "
        "require the API key."
    )

# Display previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input("Ask something about your uploaded documents...")

if question:
    if st.session_state.index is None:
        st.warning("Please upload and process at least one document first.")
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer..."):
            retrieved = search_documents(question, TOP_K)
            answer = generate_answer(question, retrieved)

        st.markdown(answer)

        if retrieved:
            with st.expander("🔎 Retrieved Sources"):
                for _, metadata, score in retrieved:
                    st.write(
                        f"**{metadata['source']}** — "
                        f"chunk {metadata['chunk']} — "
                        f"similarity `{score:.3f}`"
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
