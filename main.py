# we need to have 3 retrievers, BM25, Embeding and hybrid
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from retrievers.bm25_retriever import BM25Retriever
from retrievers.embedding_retriever import EmbeddingRetriever
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.rerank_retriever import RerankRetriever
from retrievers.rrf_retriever import RRFRetriever
from retrievers.tfidf_retriever import TfidfRetriever

HERE = Path(__file__).parent
K = 3
# Indices in the retrievers dict where a new family starts (keyword | dense | combined),
# used to draw separators on the chart. Update if you reorder or add retrievers.
GROUP_BREAKS = (3, 5)


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
    results = {}
    for name, retriever in retrievers.items():
        print(f"== {name.replace(chr(10), ' ')}")
        ranks = []
        for question, gold in zip(questions, GOLD):
            # Full ranking, so we know where the gold passage landed even outside the top K.
            # (RerankRetriever ties everything outside its shortlist at -inf; such ranks are
            # > 20 either way, so they barely move MRR.)
            ranking = retriever.retrieve(question, k=len(passages))
            rank = ranking.index(gold) + 1
            ranks.append(rank)
            print(f"  [{'HIT ' if rank <= K else 'MISS'}] rank={rank:<3} top{K}={ranking[:K]} gold={gold}  {question}")
        ranks = np.array(ranks)
        results[name] = {
            "hit@1": np.mean(ranks == 1),
            f"hit@{K}": np.mean(ranks <= K),
            "MRR": np.mean(1 / ranks),
        }
        r = results[name]
        print(f"  hit@1 = {r['hit@1']:.0%}  hit@{K} = {r[f'hit@{K}']:.0%}  MRR = {r['MRR']:.2f}\n")

    # Grouped bar chart: one group per retriever, one bar per metric.
    metrics = list(next(iter(results.values())))
    x = np.arange(len(results))
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (metric, color) in enumerate(zip(metrics, ["#8DA0CB", "#4C72B0", "#E5862D"])):
        values = [r[metric] for r in results.values()]
        bars = ax.bar(x + (i - 1) * width, values, width, label=metric, color=color)
        labels = [f"{v:.2f}" if metric == "MRR" else f"{v:.0%}" for v in values]
        ax.bar_label(bars, labels=labels, fontsize=7, padding=1)
    for b in GROUP_BREAKS:  # separate keyword | dense | combined
        ax.axvline(b - 0.5, color="#BBBBBB", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, results.keys())
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title("Retriever comparison")
    ax.legend(loc="upper left", ncols=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "retriever_scores.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
