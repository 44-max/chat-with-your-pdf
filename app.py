"""Chat with your PDF - a simple RAG app built with Streamlit."""
import os

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # overlap so sentences are not cut badly
TOP_K = 4             # how many chunks to retrieve


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def read_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def split_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


def retrieve(question: str, chunks, embeddings, embedder):
    q = embedder.encode([question], normalize_embeddings=True)[0]
    scores = embeddings @ q  # cosine similarity (vectors are normalized)
    top = np.argsort(scores)[::-1][:TOP_K]
    return [chunks[i] for i in top]


def ask_llm(question: str, context_chunks: list[str]) -> str:
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    if st.session_state.get("file_name") != uploaded.name:
        with st.spinner("Reading and indexing your PDF..."):
            text = read_pdf(uploaded)
            chunks = split_text(text)
            embedder = load_embedder()
            embeddings = embedder.encode(chunks, normalize_embeddings=True)
        st.session_state.update(
            file_name=uploaded.name, chunks=chunks,
            embeddings=embeddings, history=[],
        )
        st.success(f"Indexed {len(chunks)} chunks.")

    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)

    if question := st.chat_input("Ask something about your PDF"):
        st.chat_message("user").write(question)
        embedder = load_embedder()
        relevant = retrieve(
            question, st.session_state.chunks,
            st.session_state.embeddings, embedder,
        )
        if os.getenv("GROQ_API_KEY"):
            answer = ask_llm(question, relevant)
        else:
            answer = "No API key found. Showing the most relevant passages:\n\n" + "\n\n---\n\n".join(relevant)
        st.chat_message("assistant").write(answer)
        with st.expander("Sources used"):
            for c in relevant:
                st.write(c)
        st.session_state.history += [("user", question), ("assistant", answer)]
else:
    st.info("Upload a PDF to get started.")