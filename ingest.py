"""
Ingestion pipeline: read files → chunk → embed → add to FAISS → save.

Run via Streamlit's "Build / Refresh Index" button or from terminal:
    python ingest.py
"""

import os
import glob
import fitz                      # PyMuPDF
import markdown
from bs4 import BeautifulSoup

# Make HuggingFace tokenizers quieter after forks (cosmetic warning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rag_core.splitter import split_into_chunks
from rag_core.embedder import Embedder
from rag_core.index import VectorIndex

SRC_DIR = "data/source"   # where uploads/manual docs live
STORE_DIR = "data/store"  # FAISS + parquet are written here


# ---------- Readers ----------

def read_pdf(path: str) -> str:
    """Extract text from a PDF using PyMuPDF."""
    doc = fitz.open(path)
    text = " ".join(page.get_text("text") for page in doc)
    return text


def read_md(path: str) -> str:
    """Convert Markdown → HTML → plain text to strip markup cleanly."""
    with open(path, "r", encoding="utf-8") as f:
        html = markdown.markdown(f.read())
    return BeautifulSoup(html, "html.parser").get_text(" ")


def read_txt(path: str) -> str:
    """Read a text file robustly:
       - try utf-8
       - fall back to cp1252
       - last resort: utf-8 with errors='ignore'
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        pass
    try:
        with open(path, "r", encoding="cp1252") as f:
            return f.read()
    except UnicodeDecodeError:
        pass
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_text(path: str) -> str:
    """Dispatch to appropriate reader. Lightly guard against binary-ish files."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = read_pdf(path)
    elif ext in (".md", ".markdown"):
        text = read_md(path)
    else:
        text = read_txt(path)

    # Skip files that decode to very few printable characters (likely binary)
    if not text:
        raise ValueError("empty text after decoding")
    printable = sum(ch.isprintable() for ch in text)
    if printable / max(1, len(text)) < 0.7:
        raise ValueError("file appears non-text/binary after decode")
    return text


# ---------- Main loop ----------

def main():
    os.makedirs(SRC_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*")))
    if not files:
        print(f"Put PDFs/MD/TXT into {SRC_DIR} and re-run.")
        return

    # Initialize model & index once
    emb = Embedder()          # loads all-MiniLM-L6-v2 (384-dim, normalized)
    idx = VectorIndex()       # FAISS + parquet sidecar
    idx.load()                # create new if none on disk

    total_chunks_added = 0
    ingested_files = 0
    skipped_files = 0

    for di, path in enumerate(files):
        fname = os.path.basename(path)
        try:
            text = load_text(path)

            # Tune chunking here if needed
            chunks = split_into_chunks(text, chunk_size=800, overlap=120)
            if not chunks:
                print(f"⚠️  Skipping {fname}: produced 0 chunks")
                skipped_files += 1
                continue

            metas = [{
                "doc_id": di,
                "chunk_id": ci,
                "source": fname,
                "text": c
            } for ci, c in enumerate(chunks)]

            embs = emb.encode([m["text"] for m in metas])  # (n, 384) float32 normalized
            idx.add(embs, metas)

            total_chunks_added += len(chunks)
            ingested_files += 1
            print(f"✅ Ingested {fname}: {len(chunks)} chunks")

        except Exception as e:
            # Do not let one bad file kill the whole ingestion
            skipped_files += 1
            print(f"⚠️  Skipping {fname}: {e}")

    # Persist index + metadata
    idx.save()
    print("\n—— Ingestion summary ——")
    print(f" Files found     : {len(files)}")
    print(f" Ingested OK     : {ingested_files}")
    print(f" Skipped         : {skipped_files}")
    print(f" Chunks indexed  : {total_chunks_added}")
    print(f" Metadata rows   : {len(idx.meta)}")
    print(f" Store directory : {os.path.abspath(STORE_DIR)}")
    print("✅ Ingestion complete. Index saved to data/store/.")


if __name__ == "__main__":
    main()