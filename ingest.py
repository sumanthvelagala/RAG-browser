import requests
from pathlib import Path
from pypdf import PdfReader
from cleaners import clean_text
from chunking import page_aware_chunks

API_URL    = "http://127.0.0.1:8000/store"
PAPERS_DIR = Path(__file__).parent / "papers"

def extract_chunks(pdf_path: Path, token_limit=128, overlap=32):
    """Extract chunks with page numbers by processing each PDF page separately."""
    reader = PdfReader(str(pdf_path))

    all_words: list[str] = []
    page_boundaries: list[tuple[int, int]] = []  # (start_word_idx, page_num)

    for page_num, page in enumerate(reader.pages, 1):
        page_text = clean_text(page.extract_text() or "")
        words = page_text.split()
        if words:
            page_boundaries.append((len(all_words), page_num))
            all_words.extend(words)

    # returns list of (chunk_text, page_num)
    return page_aware_chunks(all_words, page_boundaries, token_limit, overlap)

def ingest(pdf_path: Path):
    print(f"\nProcessing: {pdf_path.name}")
    chunks = extract_chunks(pdf_path)
    print(f"  {len(chunks)} chunks to store")

    payload = {
        "pdf_id": pdf_path.name,
        "chunks": [{"title": "General", "chunk": c, "page": p} for c, p in chunks]
    }

    r = requests.post(API_URL, json=payload)
    if r.status_code == 200:
        print(f"  Stored in ChromaDB")
    else:
        print(f"  Failed: {r.status_code} {r.text}")

if __name__ == "__main__":
    import argparse, chromadb

    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true",
                        help="Wipe ChromaDB collection before ingesting")
    args = parser.parse_args()

    if args.fresh:
        client = chromadb.PersistentClient(path="/Users/Sumanth/Terminal/DataBase")
        try:
            client.delete_collection("rag_documents")
            print("Collection wiped.\n")
        except Exception:
            print("Collection not found, starting fresh.\n")

    pdfs = list(PAPERS_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} papers to ingest")
    for pdf in sorted(pdfs):
        ingest(pdf)
    print("\nAll papers ingested. Run: python eval.py")
