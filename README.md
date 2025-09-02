# RAG-Powered Q&A on Personal Docs (Local, Free)

This is a tiny, **fully local** Retrieval-Augmented QA system you can run in a few minutes.
It demonstrates the full RAG loop using **free components** only.

## High-level flow

1. **Ingest**: Read PDFs/Markdown/TXT → split into overlapping chunks (to keep context).
2. **Embed**: Convert chunks to vectors with a lightweight model (`all-MiniLM-L6-v2`).
3. **Index**: Store vectors in **FAISS** (fast vector search) + parquet metadata.
4. **Query**: User question → embed → retrieve top-k similar chunks.
5. **Answer**: Show evidence chunks and a simple extractive "answer" assembled from top hits.

> Abstractive answering (a generated summary) can be plugged in later with a small local model.

## Quick Start

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

Then in the UI:

1. Upload 1+ documents (or place them in `data/source/`).
2. Click **Build / Refresh Index**.
3. Ask a question and inspect the **Evidence** + **Answer** sections.

## Key design choices

- **Cosine similarity via Inner Product**: Normalize embeddings and use FAISS's `IndexFlatIP`.
- **Chunking**: Default `chunk_size=800`, `overlap=120`. Adjust for your content density.
- **Local-first**: No paid APIs. Everything runs on CPU.

## Files

```
rag-docs/
├─ app.py                 # Streamlit UI: upload → build index → query
├─ ingest.py              # Ingestion pipeline (read/clean → split → embed → index)
├─ rag_core/
│  ├─ splitter.py         # Chunking utilities (with comments)
│  ├─ embedder.py         # Wrapper around SentenceTransformer (normalized vectors)
│  ├─ index.py            # FAISS + metadata persistence
│  └─ answer.py           # Naive extractive answer composer
├─ data/source/           # Put your PDFs/MD/TXT here (UI saves uploads here too)
└─ data/store/            # FAISS index + Parquet metadata live here
```

## Next steps (nice v2s)

- Add **abstractive summarization** (e.g., `google/flan-t5-small`) over retrieved chunks.
- Highlight/cite exact phrases that matched the query.
- Add OCR for scanned PDFs (Tesseract) if needed.
