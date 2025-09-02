from typing import List
import re

def simple_normalize(text: str) -> str:
    """Basic text cleanup to reduce noise before chunking.
    - Replace non-breaking spaces with regular spaces.
    - Collapse repeated spaces/tabs.
    - Strip leading/trailing whitespace.

    This keeps chunk boundaries stable and improves embedding quality.
    """
    text = text.replace("\u00A0", " ")
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Split `text` into overlapping word chunks.

    Why overlap?
      Embeddings are computed per chunk. If a key sentence sits at a chunk boundary,
      small overlap helps the model see sufficient context in at least one chunk.

    Args:
      text: Full document text (already extracted/cleaned).
      chunk_size: Approx # of words per chunk (not tokens).
      overlap: Words shared between consecutive chunks.

    Returns:
      List[str]: chunked text segments.
    """
    text = simple_normalize(text)
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        # Take a window of words
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        # Advance by chunk_size - overlap to create overlap with next chunk
        i += chunk_size - overlap
        if i < 0:  # defensive guard; shouldn't happen
            break
    return chunks
