import numpy as np
from sentence_transformers import SentenceTransformer

from .base import Retriever


class EmbeddingRetriever(Retriever):
    def __init__(self, passages: list[str], model_name: str = "all-MiniLM-L6-v2", query_prefix: str = ""):
        # Some models (e.g. BGE) are trained with an instruction prefix on the query side only.
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name)
        self.passage_vecs = self.model.encode(passages, normalize_embeddings=True)

    def scores(self, query: str) -> np.ndarray:
        query_vec = self.model.encode(self.query_prefix + query, normalize_embeddings=True)
        return self.passage_vecs @ query_vec  # cosine similarity
