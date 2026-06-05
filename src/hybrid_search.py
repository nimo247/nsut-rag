import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
from sentence_transformers import SentenceTransformer
from embed import load_index, load_bm25_index

def normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s == 0:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def hybrid_search(
    query: str,
    index,           # FAISS index
    bm25,            # BM25 index
    chunks: list,
    model,           # SentenceTransformer
    top_k: int = 5,
    alpha: float = 0.5   # 0 = pure BM25, 1 = pure FAISS semantic
) -> list[dict]:
    """
    Hybrid retrieval: combine FAISS semantic + BM25 keyword scores.
    alpha controls the blend:
      alpha=0.7 → favor semantic
      alpha=0.3 → favor keyword
      alpha=0.5 → equal blend
    """
    n = len(chunks)

    # ── BM25 scores ──────────────────────────────────────
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)          # shape: (n,)
    bm25_norm = normalize(bm25_scores)

    # ── FAISS semantic scores ─────────────────────────────
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, n)         # search all

    # Convert L2 distances to similarity (lower distance = higher similarity)
    faiss_scores = np.zeros(n)
    for dist, idx in zip(distances[0], indices[0]):
        faiss_scores[idx] = 1 / (1 + dist)                 # convert to similarity
    faiss_norm = normalize(faiss_scores)

    # ── Combine ───────────────────────────────────────────
    hybrid_scores = alpha * faiss_norm + (1 - alpha) * bm25_norm

    # ── Top-k ─────────────────────────────────────────────
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text": chunks[idx]["text"],
            "filename": chunks[idx]["filename"],
            "chunk_index": chunks[idx]["chunk_index"],
            "score": float(hybrid_scores[idx]),
            "bm25_score": float(bm25_norm[idx]),
            "faiss_score": float(faiss_norm[idx])
        })

    return results


if __name__ == "__main__":
    print("Loading indexes...")
    index, chunks = load_index()
    bm25 = load_bm25_index()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Compare pure semantic vs hybrid on same query
    queries = [
        "Maxwell curl equation derivation",
        "boundary conditions for electric field",
        "what is divergence theorem"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        results = hybrid_search(query, index, bm25, chunks, model, top_k=3)

        for i, r in enumerate(results):
            print(f"\nRank {i+1} | Hybrid: {r['score']:.3f} | "
                  f"FAISS: {r['faiss_score']:.3f} | BM25: {r['bm25_score']:.3f}")
            print(f"{r['text'][:200]}")