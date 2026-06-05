import faiss
import numpy as np
import pickle
import os
import sys
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

sys.path.append(os.path.dirname(__file__))
from ingest import load_documents

MODEL_NAME = "all-MiniLM-L6-v2"


def build_index(data_dir: str, index_path: str = "index"):
    """Load all PDFs, embed chunks, build FAISS + BM25 index, save to disk."""

    # Step 1: Load chunks
    print("Loading documents...")
    chunks = load_documents(data_dir)
    texts = [c["text"] for c in chunks]
    print(f"Total chunks to embed: {len(texts)}")

    # Step 2: Embed
    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding chunks... (may take 1-2 mins)")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")
    print(f"Embedding shape: {embeddings.shape}")

    # Step 3: Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"\nFAISS index built: {index.ntotal} vectors")

    # Step 4: Save FAISS index + chunks
    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, f"{index_path}/faiss.index")
    with open(f"{index_path}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"FAISS index saved to {index_path}/")

    # Step 5: Build + save BM25 index
    build_bm25_index(chunks, index_path)


def build_bm25_index(chunks: list, index_path: str = "index"):
    """Build BM25 index from chunks and save to disk."""
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(f"{index_path}/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print("BM25 index saved.")
    return bm25


def load_index(index_path: str = "index"):
    """Load saved FAISS index + chunks from disk."""
    index = faiss.read_index(f"{index_path}/faiss.index")
    with open(f"{index_path}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def load_bm25_index(index_path: str = "index"):
    """Load BM25 index from disk."""
    with open(f"{index_path}/bm25.pkl", "rb") as f:
        return pickle.load(f)


def search(query: str, index, chunks: list, model, top_k: int = 5) -> list[dict]:
    """FAISS-only semantic search. Returns top_k chunks."""
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "text": chunks[idx]["text"],
            "filename": chunks[idx]["filename"],
            "chunk_index": chunks[idx]["chunk_index"],
            "score": float(dist)
        })
    return results


if __name__ == "__main__":
    build_index("data")

    print("\n--- Testing FAISS Retrieval ---")
    index, chunks = load_index()
    model = SentenceTransformer(MODEL_NAME)

    query = "What is the electric field of a point charge?"
    print(f"Query: {query}\n")

    results = search(query, index, chunks, model)
    for i, r in enumerate(results):
        print(f"Result {i+1} | File: {r['filename']} | Score: {r['score']:.2f}")
        print(f"{r['text'][:200]}\n")