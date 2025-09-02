from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    """Thin wrapper around SentenceTransformer model.

    - Uses `all-MiniLM-L6-v2` (384-dim) by default.
    - Normalizes embeddings so cosine similarity == inner product,
      allowing us to use FAISS IndexFlatIP efficiently.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Download/load the model. First run may take a minute.
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """Encode a list of strings into L2-normalized float32 vectors.

        Args:
          texts: List[str] of chunks or a single-question list.

        Returns:
          np.ndarray of shape (n, 384) dtype float32.
        """
        emb = self.model.encode(
            texts,
            normalize_embeddings=True,     # enables cosine via dot product
            convert_to_numpy=True
        )
        return emb.astype(np.float32)
