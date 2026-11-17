# we need to have 3 retrievers, BM25, Embeding and hybrid
import re
from pathlib import Path

import matplotlib.pyplot as plt

from retrievers.bm25_retriever import BM25Retriever
from retrievers.embedding_retriever import EmbeddingRetriever
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.rerank_retriever import RerankRetriever
from retrievers.rrf_retriever import RRFRetriever
from retrievers.tfidf_retriever import TfidfRetriever

HERE = Path(__file__).parent
K = 3


def load_passages() -> list[str]:
    # Split into sentences so each passage is small and the metric can tell methods apart.
    text = (HERE / "royal-family.txt").read_text(encoding="utf-8")
    text = re.sub(r"\[\d+\]", "", text)  # drop citation markers like [13]
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def load_questions(passages: list[str]) -> tuple[list[str], list[int]]:
    # Each line is "question | quote". The gold passage is the one sentence containing
    # the quote, so editing the source text doesn't break the labels.
    questions, gold = [], []
    for line in (HERE / "retriever-question.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        question, quote = (part.strip() for part in line.split("|", 1))
        matches = [i for i, p in enumerate(passages) if quote in p]
        assert len(matches) == 1, f"quote {quote!r} matches {len(matches)} passages, need exactly 1"
        questions.append(question)
        gold.append(matches[0])
    return questions, gold


def main():
    passages = load_passages()
    questions, GOLD = load_questions(passages)

    # Initialize the retrievers
    bm25 = BM25Retriever(passages)
    embedding = EmbeddingRetriever(passages)
    hybrid_rrf = RRFRetriever(BM25Retriever(passages, drop_stopwords=True), embedding)
    retrievers = {
        # keyword
        "TF-IDF": TfidfRetriever(passages),
        "BM25": bm25,
        "BM25\n-stopwords": BM25Retriever(passages, drop_stopwords=True),
        # dense
        "MiniLM": embedding,
        "BGE-small": EmbeddingRetriever(
            passages, "BAAI/bge-small-en-v1.5",
            query_prefix="Represent this sentence for searching relevant passages: "),
        # combined
        "Hybrid\n(avg)": HybridRetriever(bm25, embedding),
        "Hybrid\n(RRF)": hybrid_rrf,
        "RRF +\nrerank": RerankRetriever(passages, hybrid_rrf),
    }

    print(f"{len(passages)} passages, {len(questions)} questions\n")
    hit_rates = {}
    for name, retriever in retrievers.items():
        hits = 0
        print(f"== {name.replace(chr(10), ' ')}")
        for question, gold in zip(questions, GOLD):
            top = retriever.retrieve(question, k=K)
            hit = gold in top
            hits += hit
            print(f"  [{'HIT ' if hit else 'MISS'}] top{K}={top} gold={gold}  {question}")
        hit_rates[name] = hits / len(questions)
        print(f"  hit@{K} = {hit_rates[name]:.0%}\n")

    # Bar chart of hit@3 per method
    # Colour by family: keyword = blue, dense = green, combined = red.
    colors = ["#4C72B0"] * 3 + ["#55A868"] * 2 + ["#C44E52"] * 3
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(hit_rates.keys(), hit_rates.values(), color=colors[:len(hit_rates)])
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in hit_rates.values()])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(f"hit@{K}")
    ax.set_title(f"Retriever comparison (hit@{K})")
    fig.tight_layout()
    fig.savefig(HERE / "hit_at_3.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
