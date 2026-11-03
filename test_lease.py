"""
End-to-end pipeline test on real lease.pdf:
pypdf → clean_text → sliding_window_chunks → ChromaDB + BM25 → extractive QA
"""
import sys
sys.path.insert(0, "/Users/Sumanth/Terminal/RAG")

import numpy as np
from io import BytesIO
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from cleaners import clean_text
from chunking import sliding_window_chunks

PDF_PATH    = "/Users/Sumanth/Terminal/RAG/test_docs/lease.pdf"
CHUNK_TOKENS  = 128
CHUNK_OVERLAP = 32
RETRIEVE_K    = 50
RRF_FINAL_K   = 10
RRF_K         = 60
DEDUP_SIM     = 0.90
TOP_N         = 5
CHUNK_W       = 0.3

print("Loading embedder...")
embedder = SentenceTransformer("all-mpnet-base-v2")

# ── 1. Ingest PDF ─────────────────────────────────────────────────────────────
print("Ingesting lease.pdf...")
raw = ""
for page in PdfReader(PDF_PATH).pages:
    raw += (page.extract_text() or "") + "\n"

text   = clean_text(raw)
chunks = sliding_window_chunks(text, CHUNK_TOKENS, CHUNK_OVERLAP)
print(f"  {len(chunks)} chunks after sliding window")
for i, c in enumerate(chunks):
    print(f"  chunk[{i}]: {c[:100]!r}...")
print()

# ── 2. Index into ChromaDB + BM25 ────────────────────────────────────────────
client     = chromadb.Client()
collection = client.create_collection("lease_test")

print("Encoding and indexing chunks...")
embs = embedder.encode(chunks, normalize_embeddings=True)
for i, (chunk, emb) in enumerate(zip(chunks, embs)):
    collection.add(
        ids=[f"chunk_{i}"],
        documents=[chunk],
        embeddings=[emb.tolist()],
        metadatas=[{"pdf_id": "lease.pdf"}],
    )

bm25     = BM25Okapi([c.lower().split() for c in chunks])
bm25_ids = [f"chunk_{i}" for i in range(len(chunks))]
print(f"  Indexed {len(chunks)} chunks\n")


# ── 3. Hybrid retrieval ───────────────────────────────────────────────────────
def hybrid_search(query):
    k     = min(RETRIEVE_K, len(chunks))
    q_emb = embedder.encode(query, normalize_embeddings=True)

    results     = collection.query(query_embeddings=[q_emb.tolist()], n_results=k)
    dense_ids   = results["ids"][0]
    dense_docs  = results["documents"][0]
    dense_metas = results["metadatas"][0]

    bm25_scores = bm25.get_scores(query.lower().split())
    top_bm25    = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]

    fused = {}
    for rank, (did, doc, meta) in enumerate(zip(dense_ids, dense_docs, dense_metas)):
        fused.setdefault(did, {"score": 0.0, "doc": doc, "meta": meta})
        fused[did]["score"] += 2.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(top_bm25):
        did = bm25_ids[idx]
        if did not in fused:
            fused[did] = {"score": 0.0, "doc": chunks[idx], "meta": {"pdf_id": "lease.pdf"}}
        fused[did]["score"] += 1.0 / (RRF_K + rank + 1)

    return sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:RRF_FINAL_K]


# ── 4. Extractive QA ──────────────────────────────────────────────────────────
def extract_answer(query, results):
    n = len(results)
    sentences, sources, weights = [], [], []
    for rank, r in enumerate(results):
        w = 1.0 - (rank / n) * CHUNK_W
        for s in r["doc"].split("."):
            s = s.strip()
            if len(s) > 20:
                sentences.append(s)
                sources.append(r["meta"].get("pdf_id", "unknown"))
                weights.append(w)

    if not sentences:
        return "Not found in the document."

    q_emb      = embedder.encode(query, normalize_embeddings=True)
    s_embs     = embedder.encode(sentences, normalize_embeddings=True)
    sim_scores = s_embs @ q_emb

    if float(sim_scores.max()) < 0.50:
        return "Not found in the document."

    scores = sim_scores * np.array(weights)

    ranked = scores.argsort()[::-1]
    selected, sel_embs = [], []
    for idx in ranked:
        if len(selected) >= TOP_N:
            break
        emb = s_embs[idx]
        if sel_embs and max(float(emb @ se) for se in sel_embs) > DEDUP_SIM:
            continue
        selected.append((sentences[idx], sources[idx]))
        sel_embs.append(emb)

    return "\n".join(f"  • {s} [{src}]" for s, src in selected)


# ── 5. Test questions — diverse and complex ───────────────────────────────────
QUESTIONS = [
    # Simple factual
    "What is the monthly rent?",
    "What is the late fee?",
    "What is the security deposit amount?",
    # Negation / permission
    "Are pets allowed?",
    "Is smoking permitted on the property?",
    "Can the tenant sublet without permission?",
    # Lists / responsibilities
    "What utilities does the tenant pay?",
    "What repairs is the landlord responsible for?",
    # Time / process
    "How much notice is needed to terminate the lease?",
    "How long does the landlord have to return the security deposit?",
    "How much notice must the landlord give before entering?",
    # Complex / multi-condition
    "What are all the things that are grounds for eviction?",
    "What happens if the tenant wants to leave before the lease ends?",
    "If rent is paid late, what is the total amount owed on the 7th?",
    "What are all the tenant's financial obligations beyond monthly rent?",
]

print("=" * 70)
print("FULL PIPELINE TEST — lease.pdf")
print("=" * 70)
for i, q in enumerate(QUESTIONS, 1):
    results = hybrid_search(q)
    answer  = extract_answer(q, results)
    print(f"\nQ{i:02d}: {q}")
    print(answer)
print("\n" + "=" * 70)
