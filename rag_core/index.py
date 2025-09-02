import faiss, numpy as np, pandas as pd, os
from typing import List, Dict, Tuple

class VectorIndex:
    """FAISS index + sidecar metadata in Parquet.

    Store:
      - FAISS index (vectors only) in `faiss.index`
      - metadata (doc_id, chunk_id, source filename, text) in `meta.parquet`

    The metadata helps display snippets & filenames when retrieved by vector.
    """
    def __init__(self, dim: int = 384, store_dir: str = "data/store"):
        self.dim = dim
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)
        self.index_path = os.path.join(store_dir, "faiss.index")
        self.meta_path  = os.path.join(store_dir, "meta.parquet")
        self.index = None
        # Initialize empty metadata table
        self.meta = pd.DataFrame(columns=["doc_id","chunk_id","source","text"])

    def _new_index(self):
        """Create an Inner Product index.

        Because embeddings are unit-normalized, inner product == cosine similarity.
        """
        idx = faiss.IndexFlatIP(self.dim)
        return idx

    def load(self):
        """Load index + metadata from disk if present, else create fresh index."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = self._new_index()
        if os.path.exists(self.meta_path):
            self.meta = pd.read_parquet(self.meta_path)

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
        """Append vectors + metadata to the index.

        Args:
          embeddings: shape (n, dim) float32, **already normalized**
          metadatas:  list of dicts parallel to embeddings rows
        """
        if self.index is None:
            self.load()
        self.index.add(embeddings)
        self.meta = pd.concat([self.meta, pd.DataFrame(metadatas)], ignore_index=True)

    def save(self):
        """Persist both FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        self.meta.to_parquet(self.meta_path, index=False)

    def search(self, query_emb: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Search top-k nearest neighbors for a single query vector.

        Args:
          query_emb: shape (1, dim) vector for the user question (normalized).
          k: how many candidates to return.

        Returns:
          (D, I, rows) where:
            D: similarity scores (cosine via inner product)
            I: indices in the FAISS index
            rows: list of metadata dicts for each retrieved vector
        """
        if self.index is None:
            self.load()
        D, I = self.index.search(query_emb.astype(np.float32), k)
        rows = []
        for idx in I[0]:
            if idx == -1:
                continue
            rows.append(self.meta.iloc[int(idx)].to_dict())
        return D[0], I[0], rows
