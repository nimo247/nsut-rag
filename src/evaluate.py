import os
import sys
sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from embed import load_index, load_bm25_index
from retrieve import ask
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_utilization
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq

load_dotenv()

# ── Test questions about your PDF ────────────────────────
# Write 5 questions you know the answers to from your document
EVAL_QUESTIONS = [
    "What are Maxwell's equations in differential form?",
    "What is Gauss's law for electric fields?",
    "What is the relationship between electric field and electric potential?",
    "State Faraday's law of induction.",
    "What is the continuity equation in electromagnetics?",
]

def run_evaluation():
    print("Loading indexes...")
    index, chunks = load_index()
    bm25 = load_bm25_index()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Collect RAG outputs
    questions = []
    answers = []
    contexts = []

    print("Running RAG pipeline on eval questions...\n")
    for q in EVAL_QUESTIONS:
        print(f"Q: {q}")
        result = ask(q, index, chunks, model, bm25=bm25, top_k=5)
        questions.append(q)
        answers.append(result["answer"])
        contexts.append([s["filename"] + ": " + 
                        chunks[s["chunk_index"]]["text"] 
                        for s in result["sources"]])
        print(f"A: {result['answer'][:100]}...\n")

    # Build dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    })

    # Use Groq as evaluation LLM
    groq_llm = LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        )
    )

    print("Running RAGAS evaluation...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_utilization],
        llm=groq_llm,
    )

    print("\n" + "="*50)
    print("RAGAS EVALUATION RESULTS")
    print("="*50)
    df = results.to_pandas()
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision"]].to_string())
    print("\nMean scores:")
    print(f"  Faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"  Answer Relevancy:  {df['answer_relevancy'].mean():.3f}")
    print(f"  Context Utilization: {df['context_utilization'].mean():.3f}")

    # Save results
    df.to_csv("evaluation_results.csv", index=False)
    print("\nSaved to evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()