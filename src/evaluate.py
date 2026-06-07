import os
import sys
sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from embed import load_index, load_bm25_index
from retrieve import ask
from groq import Groq
import numpy as np
import pandas as pd

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"

EVAL_QUESTIONS = [
    "What are Maxwell's equations in differential form?",
    "What is Gauss's law for electric fields?",
    "What is the relationship between electric field and electric potential?",
    "State Faraday's law of induction.",
    "What is the continuity equation in electromagnetics?",
]

# Reference answers (ground truth)
GROUND_TRUTHS = [
    "Maxwell's equations are: ∇·D = ρv, ∇·B = 0, ∇×E = -∂B/∂t, ∇×H = J + ∂D/∂t",
    "Gauss's law states that the total electric flux through a closed surface equals the enclosed charge: ∮D·dS = Q_enc",
    "The electric field is the negative gradient of electric potential: E = -∇V",
    "Faraday's law states that the induced EMF equals the negative rate of change of magnetic flux: ∮E·dl = -dΦ/dt",
    "The continuity equation is: ∇·J + ∂ρv/∂t = 0, expressing conservation of charge",
]


def score_with_llm(question: str, answer: str, context: str, ground_truth: str) -> dict:
    """Use Groq to score faithfulness and relevancy."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are evaluating a RAG system. Score the following on a scale of 0.0 to 1.0.

QUESTION: {question}
GROUND TRUTH: {ground_truth}
RETRIEVED CONTEXT: {context[:500]}
GENERATED ANSWER: {answer[:500]}

Score these three metrics:
1. Faithfulness (0-1): Is the answer grounded in the retrieved context? No hallucinations?
2. Answer Relevancy (0-1): Does the answer actually address the question?
3. Context Utilization (0-1): Did the answer make good use of the retrieved context?

Respond ONLY in this exact format:
faithfulness: <score>
answer_relevancy: <score>
context_utilization: <score>"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    text = response.choices[0].message.content
    scores = {}
    for line in text.strip().split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            try:
                scores[key.strip()] = float(val.strip())
            except:
                scores[key.strip()] = 0.0
    return scores


def run_evaluation():
    print("Loading indexes...")
    index, chunks = load_index()
    bm25 = load_bm25_index()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    results = []

    print("Running evaluation...\n")
    for q, gt in zip(EVAL_QUESTIONS, GROUND_TRUTHS):
        print(f"Q: {q}")
        result = ask(q, index, chunks, model, bm25=bm25, top_k=5)
        answer = result["answer"]
        context = " ".join([chunks[s["chunk_index"]]["text"] for s in result["sources"]])

        scores = score_with_llm(q, answer, context, gt)
        scores["question"] = q
        results.append(scores)

        print(f"  Faithfulness: {scores.get('faithfulness', 0):.2f} | "
              f"Relevancy: {scores.get('answer_relevancy', 0):.2f} | "
              f"Context Util: {scores.get('context_utilization', 0):.2f}\n")

    df = pd.DataFrame(results)

    print("=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    print(df[["question", "faithfulness", "answer_relevancy", "context_utilization"]].to_string())
    print("\nMean scores:")
    print(f"  Faithfulness:        {df['faithfulness'].mean():.3f}")
    print(f"  Answer Relevancy:    {df['answer_relevancy'].mean():.3f}")
    print(f"  Context Utilization: {df['context_utilization'].mean():.3f}")

    df.to_csv("evaluation_results.csv", index=False)
    print("\nSaved to evaluation_results.csv")


if __name__ == "__main__":
    run_evaluation()