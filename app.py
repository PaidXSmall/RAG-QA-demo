"""Streamlit UI for the local RAG demo.

Usage:
    streamlit run app.py
"""
import streamlit as st
from rag_core.embedder import Embedder
from rag_core.index import VectorIndex
from rag_core.answer import compose_extractive_answer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Basic Streamlit page setup
st.set_page_config(page_title="RAG Q&A (Local, Free)", layout="wide")
st.title("🔎 RAG-Powered Q&A on Personal Docs")

st.markdown(
    "1) Upload docs → 2) Click **Build Index** → 3) Ask questions. "
)

# --- File upload ---
uploaded = st.file_uploader(
    "Upload PDF / Markdown / TXT (you can select multiple files)",
    type=["pdf","md","markdown","txt"],
    accept_multiple_files=True
)

# Save uploads to data/source so the ingestion script can find them
if uploaded:
    import os
    os.makedirs("data/source", exist_ok=True)
    for f in uploaded:
        with open(f"data/source/{f.name}", "wb") as out:
            out.write(f.getbuffer())
    st.success(f"Saved {len(uploaded)} file(s) to data/source/.")

# --- Build index button ---
if st.button("📦 Build / Refresh Index"):
    import subprocess, sys
    st.info("Building index (chunk → embed → index)…")
    ret = subprocess.run([sys.executable, "ingest.py"])
    if ret.returncode == 0:
        st.success("Index ready! Ask away below.")

# --- Query section ---
query = st.text_input("Your question")
topk = st.slider("Top-K passages", min_value=3, max_value=10, value=5)

if st.button("🔍 Search") and query:
    # Lazily load the embedder & index for querying
    emb = Embedder()
    idx = VectorIndex(); idx.load()

    # Encode the user query and retrieve similar chunks
    q_emb = emb.encode([query])           # shape (1, 384)
    D, I, rows = idx.search(q_emb, k=topk)

    # Evidence panel: each retrieved chunk with its similarity score
    st.subheader("Evidence")
    for score, row in zip(D, rows):
        with st.expander(f"{row['source']}  |  score={float(score):.3f}"):
            st.write(row["text"])

    # Answer panel: a very simple extractive result based on the hits
    st.subheader("Answer")
    st.code(compose_extractive_answer(query, rows), language="markdown")
