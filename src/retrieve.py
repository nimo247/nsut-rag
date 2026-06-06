import os
import sys
sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from embed import load_index, load_bm25_index
from hybrid_search import hybrid_search

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"
MODEL_NAME = "all-MiniLM-L6-v2"


def build_prompt(query: str, chunks: list, chat_history: list = []) -> str:
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Source {i+1} - {chunk['filename']}]\n{chunk['text']}\n\n"

    # Build last 3 exchanges from history
    history_str = ""
    if chat_history:
        history_str = "PREVIOUS CONVERSATION:\n"
        for msg in chat_history[-6:]:  # last 3 Q&A pairs
            role = "Student" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
        history_str += "\n"

    return f"""You are a helpful study assistant for engineering students.
Using the context below, give a clear, well-explained answer to the question.
- Explain concepts in your own words, don't just copy text from the source
- Use the equations and facts from the context to support your explanation
- If the context contains relevant equations, include and explain them
- If the context truly has no relevant information, say so
- Always mention which source(s) you used

{history_str}CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


def ask(query: str, index, chunks: list, model, bm25=None,
        top_k: int = 5, alpha: float = 0.5, chat_history: list = []) -> dict:
    """Full RAG pipeline: hybrid retrieve + generate."""

    if bm25 is not None:
        results = hybrid_search(query, index, bm25, chunks, model, top_k=top_k, alpha=alpha)
    else:
        from embed import search
        results = search(query, index, chunks, model, top_k=top_k)

    prompt = build_prompt(query, results, chat_history)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return {
        "query": query,
        "answer": response.choices[0].message.content,
        "sources": [
            {
                "filename": r["filename"],
                "chunk_index": r["chunk_index"],
                "score": r["score"],
                "bm25_score": r.get("bm25_score"),
                "faiss_score": r.get("faiss_score")
            }
            for r in results
        ]
    }


if __name__ == "__main__":
    print("Loading indexes...")
    index, chunks = load_index()
    bm25 = load_bm25_index()
    model = SentenceTransformer(MODEL_NAME)

    queries = [
        "What is Gauss's law and when do we use it?",
        "Explain the concept of electric potential.",
        "What is the difference between electric field and magnetic field?"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        print('='*60)
        result = ask(query, index, chunks, model, bm25=bm25)
        print(f"A: {result['answer']}")
        print(f"\nSources:")
        for s in result['sources'][:2]:
            print(f"  - chunk {s['chunk_index']} | hybrid: {s['score']:.2f} | faiss: {s.get('faiss_score', 0):.2f} | bm25: {s.get('bm25_score', 0):.2f}")