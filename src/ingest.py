import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    chunk_size: number of words per chunk
    overlap: number of words shared between consecutive chunks
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap ensures context isn't lost at boundaries

    return chunks


def load_documents(data_dir: str) -> list[dict]:
    """
    Load all PDFs from a directory.
    Returns list of {filename, chunk_index, text}
    """
    all_chunks = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(data_dir, filename)
            print(f"Processing: {filename}")

            text = extract_text_from_pdf(path)
            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "filename": filename,
                    "chunk_index": i,
                    "text": chunk
                })

            print(f"  → {len(chunks)} chunks created")

    return all_chunks


# Quick test
if __name__ == "__main__":
    chunks = load_documents("data")
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"\nSample chunk:\n{chunks[0]['text'][:300]}")