import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from embed import load_index, build_index
from retrieve import ask

load_dotenv()

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="NSUT Study Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 NSUT Study Assistant")
st.caption("Ask questions from your uploaded course notes")

# ── Load index + model (cached so it doesn't reload every query) ──
@st.cache_resource
def load_resources():
    index, chunks = load_index()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, model

# ── Sidebar: upload + reindex ─────────────────────────────
with st.sidebar:
    st.header("📁 Documents")

    uploaded = st.file_uploader(
        "Upload PDF notes", 
        type="pdf", 
        accept_multiple_files=True
    )

    if uploaded:
        os.makedirs("data", exist_ok=True)
        for f in uploaded:
            with open(f"data/{f.name}", "wb") as out:
                out.write(f.read())
        st.success(f"{len(uploaded)} file(s) saved to data/")

    if st.button("🔄 Rebuild Index"):
        with st.spinner("Rebuilding index..."):
            build_index("data")
            st.cache_resource.clear()
        st.success("Index rebuilt!")
        st.rerun()

    st.divider()
    st.markdown("**Indexed files:**")
    if os.path.exists("data"):
        pdfs = [f for f in os.listdir("data") if f.endswith(".pdf")]
        for pdf in pdfs:
            st.markdown(f"- {pdf}")
    
    top_k = st.slider("Chunks to retrieve", min_value=3, max_value=10, value=5)

# ── Main chat interface ───────────────────────────────────
if "messages" in st.session_state and len(st.session_state.messages) == 0:
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s['filename']}` — chunk {s['chunk_index']} (score: {s['score']:.2f})")

# Query input
query = st.chat_input("Ask something from your notes...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching notes..."):
            try:
                index, chunks, model = load_resources()
                result = ask(query, index, chunks, model, top_k=top_k)
                st.markdown(result["answer"])

                with st.expander("📎 Sources"):
                    for s in result["sources"]:
                        st.markdown(f"- `{s['filename']}` — chunk {s['chunk_index']} (score: {s['score']:.2f})")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except Exception as e:
                st.error(f"Error: {e}")