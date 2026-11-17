import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import Retriever


class TfidfRetriever(Retriever):
    """Classic keyword baseline: TF-IDF vectors compared with cosine similarity."""

    def __init__(self, passages: list[str]):
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.passage_vecs = self.vectorizer.fit_transform(passages)  # rows are L2-normalised

    def scores(self, query: str) -> np.ndarray:
        return (self.passage_vecs @ self.vectorizer.transform([query]).T).toarray().ravel()
