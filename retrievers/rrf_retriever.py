import numpy as np

from .base import Retriever


class RRFRetriever(Retriever):
    """Reciprocal rank fusion: combine rankings instead of scores.

    Each passage gets sum(1 / (k + rank)) over the retrievers. Only rank order matters,
    so one retriever's noisy or oddly scaled scores can't swamp the other.
    """

    def __init__(self, *retrievers: Retriever, k: int = 60):
        self.retrievers, self.k = retrievers, k

    def scores(self, query: str) -> np.ndarray:
        total = None
        for r in self.retrievers:
            order = np.argsort(r.scores(query))[::-1]
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            fused = 1.0 / (self.k + ranks)
            total = fused if total is None else total + fused
        return total
