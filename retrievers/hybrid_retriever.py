import numpy as np

from .base import Retriever


def min_max(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x, dtype=float)


class HybridRetriever(Retriever):
    """Averages the min-max normalised scores of two retrievers."""

    def __init__(self, a: Retriever, b: Retriever):
        self.a, self.b = a, b

    def scores(self, query: str) -> np.ndarray:
        return (min_max(self.a.scores(query)) + min_max(self.b.scores(query))) / 2
