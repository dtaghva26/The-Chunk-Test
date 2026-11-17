import numpy as np


class Retriever:
    """Scores every passage for a query; retrieve() returns the top-k passage indices."""

    def scores(self, query: str) -> np.ndarray:
        raise NotImplementedError

    def retrieve(self, query: str, k: int = 3) -> list[int]:
        return np.argsort(self.scores(query))[::-1][:k].tolist()
