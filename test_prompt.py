"""Full pipeline test: hybrid retrieval (BM25 + dense RRF) → extractive QA."""
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

embedder = SentenceTransformer("all-mpnet-base-v2")

# ── Corpus (simulates a full lease PDF chunked with sliding window) ───────────
CORPUS = [
    "SECTION 1. PARTIES This lease is between John Smith (Landlord) and Jane Doe (Tenant) for the property at 42 Maple Street, Austin, TX 78701.",
    "SECTION 2. TERM Lease begins January 1, 2024 and ends December 31, 2024. Month-to-month after expiration unless either party provides written notice.",
    "SECTION 3. RENT Monthly rent is $1,850 due on the 1st of each month. Grace period: 5 days. Late fee of $75 after the 6th.",
    "SECTION 4. SECURITY DEPOSIT Security deposit: $3,700 (two months rent). Returned within 30 days of termination minus damage deductions.",
    "SECTION 5. UTILITIES Tenant pays: electricity, internet, cable. Landlord pays: water, trash, gas. Accounts must be transferred within 7 days of move-in.",
    "SECTION 6. PETS No pets without written consent. Pet fee: $300 non-refundable. Only cats and dogs under 25 pounds permitted.",
    "SECTION 7. MAINTENANCE Tenant: minor repairs under $75. Report issues within 48 hours. Landlord: plumbing, heating, structural repairs.",
    "SECTION 8. ENTRY Landlord must give 24 hours notice before entry except in emergencies. Entry permitted between 8am and 6pm.",
    "SECTION 9. NOTICE OF TERMINATION 60 days written notice required. Via certified mail or hand delivery.",
    "SECTION 10. SUBLETTING No subletting without written landlord approval. Unauthorized subletting is grounds for eviction.",
    "SECTION 11. SMOKING No smoking anywhere on the property including balconies and common areas. Violation is grounds for eviction.",
    "SECTION 12. ALTERATIONS No alterations, painting, or modifications without written consent. Tenant must restore to original condition at move-out.",
    "SECTION 13. RENEWAL Lease may be renewed with 30 days notice before end of term. Rent may increase up to 5% on renewal.",
    "SECTION 14. EARLY TERMINATION Early termination fee: 2 months rent ($3,700). Must give 30 days notice. Tenant loses security deposit.",
    "SECTION 15. GOVERNING LAW This lease is governed by the laws of the State of Texas.",
]

RETRIEVE_K  = 10   # candidates per retriever
RRF_K       = 60
RRF_FINAL_K = 10   # chunks passed to extractive QA
DEDUP_SIM   = 0.90
TOP_N       = 5
CHUNK_W     = 0.3

# ── Pre-compute corpus embeddings and BM25 ────────────────────────────────────
print("Encoding corpus...")
corpus_embs = embedder.encode(CORPUS, normalize_embeddings=True)
bm25 = BM25Okapi([c.lower().split() for c in CORPUS])
print(f"Ready. {len(CORPUS)} chunks.\n")


# ── Hybrid retrieval (BM25 + dense RRF, 2:1 weight) ──────────────────────────
def hybrid_search(query):
    k     = min(RETRIEVE_K, len(CORPUS))
    q_emb = embedder.encode(query, normalize_embeddings=True)

    # dense
    sims      = corpus_embs @ q_emb
    dense_top = sims.argsort()[::-1][:k]

    # BM25
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top    = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]

    # RRF fusion (dense 2x)
    fused = {}
    for rank, idx in enumerate(dense_top):
        fused.setdefault(idx, 0.0)
        fused[idx] += 2.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(bm25_top):
        fused.setdefault(idx, 0.0)
        fused[idx] += 1.0 / (RRF_K + rank + 1)

    ranked = sorted(fused, key=fused.get, reverse=True)[:RRF_FINAL_K]
    return [(CORPUS[i], "lease.pdf") for i in ranked]   # (doc, source), ordered by RRF score


# ── Extractive QA with chunk-weighted scoring ─────────────────────────────────
def extract_answer(query, chunks):
    n = len(chunks)
    sentences, sources, weights = [], [], []
    for rank, (doc, src) in enumerate(chunks):
        w = 1.0 - (rank / n) * CHUNK_W
        for s in doc.split("."):
            s = s.strip()
            if len(s) > 20:
                sentences.append(s)
                sources.append(src)
                weights.append(w)

    if not sentences:
        return "Not found in the document."

    q_emb  = embedder.encode(query, normalize_embeddings=True)
    s_embs = embedder.encode(sentences, normalize_embeddings=True)
    scores = (s_embs @ q_emb) * np.array(weights)

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


# ── Test questions with ground-truth keywords ─────────────────────────────────
QUESTIONS = [
    ("What is the monthly rent?",                              "1,850"),
    ("What is the late fee for rent?",                         "75"),
    ("How large is the security deposit?",                     "3,700"),
    ("Are pets allowed?",                                      "written consent"),
    ("Is smoking allowed on the balcony?",                     "No smoking"),
    ("Can the tenant sublet the apartment?",                   "written landlord approval"),
    ("What utilities does the tenant pay?",                    "electricity"),
    ("What utilities does the landlord pay?",                  "water"),
    ("What repairs is the tenant responsible for?",           "minor repairs"),
    ("What repairs is the landlord responsible for?",         "plumbing"),
    ("How much notice is required to terminate the lease?",   "60 days"),
    ("How much notice must the landlord give before entry?",  "24 hours"),
    ("How long does the landlord have to return the deposit?","30 days"),
    ("What is the penalty for early termination?",            "3,700"),
    ("What happens if tenant makes alterations without permission?", "restore to original"),
]

correct = 0
for i, (q, keyword) in enumerate(QUESTIONS, 1):
    chunks = hybrid_search(q)
    answer = extract_answer(q, chunks)
    hit    = keyword.lower() in answer.lower()
    correct += hit
    marker = "✓" if hit else "✗"
    print(f"Q{i:02d} [{marker}] {q}")
    print(answer)
    print()

print(f"Score: {correct}/{len(QUESTIONS)} ({100*correct//len(QUESTIONS)}%)")
