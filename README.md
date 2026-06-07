# 📚 NSUT Study Assistant — RAG Pipeline

A Retrieval-Augmented Generation (RAG) system for querying NSUT course material using natural language. Built from scratch without LangChain.

---

## Architecture

```
PDF Ingestion (PyMuPDF)
       ↓
Chunking (500 words, 50-word overlap)
       ↓
Embeddings (sentence-transformers/all-MiniLM-L6-v2)
       ↓
Hybrid Search (FAISS semantic + BM25 keyword, α=0.5)
       ↓
LLM Answer Generation (Groq / LLaMA 3.1-8B)
       ↓
Streamlit UI with conversation memory
```

---

## Features

- **Hybrid Retrieval** — combines dense vector search (FAISS) and sparse keyword search (BM25) for better coverage
- **Conversation Memory** — follow-up questions understand prior context
- **LaTeX Rendering** — equations render properly for math-heavy course content
- **Custom Evaluation** — LLM-based scoring for faithfulness, answer relevancy, and context utilization
- **PDF Upload UI** — upload new documents and rebuild index from the Streamlit interface

---

## Evaluation Results

Evaluated on 5 questions from Sadiku's Elements of Electromagnetics.

| Metric | Score |
|---|---|
| Faithfulness | 0.900 |
| Answer Relevancy | 1.000 |
| Context Utilization | 0.280 |

> **Note:** Context utilization is lower because the LLM (LLaMA 3.1-8B) has prior knowledge of electromagnetics and blends it with retrieved context rather than citing chunks explicitly. Faithfulness and relevancy remain high, confirming the pipeline is grounded and accurate.

---

## Tech Stack

| Component | Tool |
|---|---|
| PDF Extraction | `PyMuPDF` |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Search | `FAISS` |
| Keyword Search | `rank-bm25` |
| LLM Inference | `Groq API` (LLaMA 3.1-8B) |
| UI | `Streamlit` |

---

## Project Structure

```
rag-nsut/
├── data/                  # PDF documents (add your notes here)
├── index/                 # FAISS + BM25 indexes (auto-generated)
├── src/
│   ├── ingest.py          # PDF parsing + chunking
│   ├── embed.py           # Embeddings + FAISS + BM25 index builder
│   ├── hybrid_search.py   # Hybrid retrieval logic
│   ├── retrieve.py        # Full RAG pipeline + Groq LLM
│   └── evaluate.py        # Custom evaluation metrics
├── app.py                 # Streamlit UI
├── requirements.txt
└── .env                   # API keys (not committed)
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/rag-nsut.git
cd rag-nsut

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Add API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com)

### 3. Add your PDFs

Drop any text-based PDF into the `data/` folder.

> **Note:** Scanned/image PDFs won't work — use digital-native PDFs (NPTEL, professor slides, textbooks).

### 4. Build the index

```bash
python src/embed.py
```

This builds both the FAISS vector index and BM25 keyword index.

### 5. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Running Evaluation

```bash
python src/evaluate.py
```

Edit `EVAL_QUESTIONS` and `GROUND_TRUTHS` in `evaluate.py` to match your documents. Results are saved to `evaluation_results.csv`.

---

## How It Works

### Hybrid Search

Pure semantic search (FAISS) misses exact keyword matches. Pure keyword search (BM25) misses paraphrased queries. Hybrid combines both:

```
hybrid_score = α × FAISS_score + (1 - α) × BM25_score
```

With `α = 0.5` (equal blend). Tunable via code.

### Conversation Memory

The last 3 question-answer pairs are injected into the prompt, enabling follow-up questions like:

```
Q: What are Maxwell's equations?
Q: Now explain the curl equation in more detail   ← understands "curl equation" refers to above
```

### Why Not LangChain?

Built each component from scratch — chunking, retrieval, hybrid scoring, evaluation — for full control and better understanding of the pipeline internals. Every design decision is explainable.

---

## Example Queries

```
What are Maxwell's equations in differential form?
Explain Gauss's law and when to use it
What is the difference between electric field and magnetic field?
Derive the continuity equation
What is VWAP and how is it calculated?
```

---

## .gitignore

```
venv/
.env
data/
index/
__pycache__/
*.pyc
evaluation_results.csv
```
