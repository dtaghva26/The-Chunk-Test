import numpy as np
from sentence_transformers import CrossEncoder

from .base import Retriever


class RerankRetriever(Retriever):
    """Two-stage retrieval: take the first stage's top candidates, then re-score each
    (query, passage) pair with a cross-encoder that reads both texts together."""

    def __init__(self, passages: list[str], first_stage: Retriever,
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", candidates: int = 20):
        self.passages, self.first_stage, self.candidates = passages, first_stage, candidates
        self.model = CrossEncoder(model_name)

    def scores(self, query: str) -> np.ndarray:
        top = self.first_stage.retrieve(query, k=self.candidates)
        out = np.full(len(self.passages), -np.inf)  # passages outside the shortlist rank last
        out[top] = self.model.predict([(query, self.passages[i]) for i in top])
        return out
