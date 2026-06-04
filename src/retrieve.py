import os
import sys
sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from embed import load_index, search

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"  # free, fast


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Build prompt with retrieved context."""
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Source {i+1} - {chunk['filename']}]\n{chunk['text']}\n\n"

    prompt = f"""You are a helpful study assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I couldn't find this in the provided notes."
Always mention which source you used.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt


def ask(query: str, index, chunks: list, model, top_k: int = 5) -> dict:
    """Full RAG pipeline: retrieve + generate."""

    # Step 1: Retrieve relevant chunks
    results = search(query, index, chunks, model, top_k=top_k)

    # Step 2: Build prompt
    prompt = build_prompt(query, results)

    # Step 3: Call Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # low temp = more factual
    )

    answer = response.choices[0].message.content

    return {
        "query": query,
        "answer": answer,
        "sources": [
            {"filename": r["filename"], "chunk_index": r["chunk_index"], "score": r["score"]}
            for r in results
        ]
    }


if __name__ == "__main__":
    print("Loading index and model...")
    index, chunks = load_index()
    model = SentenceTransformer(MODEL_NAME)

    # Test queries
    queries = [
        "What is Gauss's law and when do we use it?",
        "Explain the concept of electric potential.",
        "What is the difference between electric field and magnetic field?"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        print('='*60)
        result = ask(query, index, chunks, model)
        print(f"A: {result['answer']}")
        print(f"\nSources used:")
        for s in result['sources'][:2]:
            print(f"  - {s['filename']} (chunk {s['chunk_index']}, score {s['score']:.2f})")