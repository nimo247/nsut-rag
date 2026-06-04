import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
import sys
sys.path.append(os.path.dirname(__file__))
from ingest import load_documents

# Load embedding model (downloads once, ~90MB)
MODEL_NAME = "all-MiniLM-L6-v2"

def build_index(data_dir: str, index_path: str = "index"):
    """
    Load all PDFs, embed chunks, build FAISS index, save to disk.
    """
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
    print(f"Embedding shape: {embeddings.shape}")  # (406, 384)

    # Step 3: Build FAISS index
    dim = embeddings.shape[1]  # 384 for MiniLM
    index = faiss.IndexFlatL2(dim)  # L2 distance search
    index.add(embeddings)
    print(f"\nFAISS index built: {index.ntotal} vectors")

    # Step 4: Save index + metadata to disk
    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, f"{index_path}/faiss.index")
    with open(f"{index_path}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved to {index_path}/")


def load_index(index_path: str = "index"):
    """Load saved FAISS index + chunks from disk."""
    index = faiss.read_index(f"{index_path}/faiss.index")
    with open(f"{index_path}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def search(query: str, index, chunks: list, model, top_k: int = 5) -> list[dict]:
    """
    Given a query, return top_k most relevant chunks.
    """
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "text": chunks[idx]["text"],
            "filename": chunks[idx]["filename"],
            "chunk_index": chunks[idx]["chunk_index"],
            "score": float(dist)  # lower = more similar
        })
    return results


# Build + test
if __name__ == "__main__":
    # Build index
    build_index("data")

    # Test retrieval
    print("\n--- Testing Retrieval ---")
    index, chunks = load_index()
    model = SentenceTransformer(MODEL_NAME)

    query = "What is the electric field of a point charge?"
    print(f"Query: {query}\n")

    results = search(query, index, chunks, model)
    for i, r in enumerate(results):
        print(f"Result {i+1} | File: {r['filename']} | Score: {r['score']:.2f}")
        print(f"{r['text'][:200]}\n")