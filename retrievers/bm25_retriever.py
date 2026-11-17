import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .base import Retriever


def tokenize(text: str, drop_stopwords: bool = False) -> list[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS] if drop_stopwords else tokens


class BM25Retriever(Retriever):
    def __init__(self, passages: list[str], drop_stopwords: bool = False):
        self.drop_stopwords = drop_stopwords
        self.bm25 = BM25Okapi([tokenize(p, drop_stopwords) for p in passages])

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query, self.drop_stopwords)))
