"""
Full pipeline test on all 15 research papers.
Reads PDFs directly — no FastAPI needed.
Hybrid RRF retrieval → chunk-weighted extractive QA
"""
import sys, time
sys.path.insert(0, "/Users/Sumanth/Terminal/RAG")

import numpy as np
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from cleaners import clean_text
from chunking import sliding_window_chunks

PAPERS_DIR  = Path("/Users/Sumanth/Terminal/RAG/papers")
CHUNK_TOKENS  = 128
CHUNK_OVERLAP = 32
RETRIEVE_K    = 50
RRF_FINAL_K   = 10
RRF_K         = 60
DEDUP_SIM     = 0.90
TOP_N         = 5
CHUNK_W       = 0.3
MIN_SIM       = 0.50

# ── Load embedder ─────────────────────────────────────────────────────────────
print("Loading embedder...")
embedder = SentenceTransformer("all-mpnet-base-v2")

# ── Ingest all papers ─────────────────────────────────────────────────────────
print("Ingesting papers...")
client     = chromadb.Client()
collection = client.create_collection("papers_test")

all_chunks, all_metas, all_ids = [], [], []

for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
    raw = ""
    for page in PdfReader(str(pdf_path)).pages:
        raw += (page.extract_text() or "") + "\n"
    chunks = sliding_window_chunks(clean_text(raw), CHUNK_TOKENS, CHUNK_OVERLAP)
    embs   = embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    for i, (chunk, emb) in enumerate(zip(chunks, embs)):
        cid = f"{pdf_path.name}_chunk{i}"
        collection.add(ids=[cid], documents=[chunk],
                       embeddings=[emb.tolist()],
                       metadatas=[{"pdf_id": pdf_path.name}])
        all_chunks.append(chunk)
        all_metas.append({"pdf_id": pdf_path.name})
        all_ids.append(cid)
    print(f"  {pdf_path.name}: {len(chunks)} chunks")

bm25 = BM25Okapi([c.lower().split() for c in all_chunks])
print(f"\nTotal: {len(all_chunks)} chunks from {len(list(PAPERS_DIR.glob('*.pdf')))} papers\n")


# ── Hybrid retrieval ──────────────────────────────────────────────────────────
def hybrid_search(query):
    k     = min(RETRIEVE_K, len(all_chunks))
    q_emb = embedder.encode(query, normalize_embeddings=True)

    res         = collection.query(query_embeddings=[q_emb.tolist()], n_results=k)
    dense_ids   = res["ids"][0]
    dense_docs  = res["documents"][0]
    dense_metas = res["metadatas"][0]

    bm25_scores = bm25.get_scores(query.lower().split())
    top_bm25    = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]

    fused = {}
    for rank, (did, doc, meta) in enumerate(zip(dense_ids, dense_docs, dense_metas)):
        fused.setdefault(did, {"score": 0.0, "doc": doc, "meta": meta})
        fused[did]["score"] += 2.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(top_bm25):
        did = all_ids[idx]
        fused.setdefault(did, {"score": 0.0, "doc": all_chunks[idx], "meta": all_metas[idx]})
        fused[did]["score"] += 1.0 / (RRF_K + rank + 1)

    return sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:RRF_FINAL_K]


# ── Extractive QA ─────────────────────────────────────────────────────────────
def extract_answer(query, results):
    n = len(results)
    sentences, sources, weights = [], [], []
    for rank, r in enumerate(results):
        w = 1.0 - (rank / n) * CHUNK_W
        for s in r["doc"].split("."):
            s = s.strip()
            if len(s) > 20:
                sentences.append(s)
                sources.append(r["meta"]["pdf_id"])
                weights.append(w)

    if not sentences:
        return "Not found in the document."

    q_emb      = embedder.encode(query, normalize_embeddings=True)
    s_embs     = embedder.encode(sentences, normalize_embeddings=True)
    sim_scores = s_embs @ q_emb

    if float(sim_scores.max()) < MIN_SIM:
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


# ── Test questions — 1 simple + 1 complex per paper ──────────────────────────
QUESTIONS = [
    # BERT
    ("What is BERT's pre-training objective?",                         "masked",            "bert.pdf"),
    ("Why does BERT use bidirectional training instead of left-to-right?", "bidirectional", "bert.pdf"),
    # Attention Is All You Need
    ("What attention mechanism does the Transformer use?",             "multi-head",        "attention_is_all_you_need.pdf"),
    ("Why did the Transformer remove recurrence?",                     "sequential",        "attention_is_all_you_need.pdf"),
    # RAG
    ("What retriever does RAG use?",                                   "DPR",               "rag_lewis.pdf"),
    ("How does RAG combine retrieved passages with generation?",       "marginalize",       "rag_lewis.pdf"),
    # DPR
    ("How does DPR differ from BM25?",                                 "BM25",              "dpr.pdf"),
    ("What encoder does DPR use for questions and passages?",          "BERT",              "dpr.pdf"),
    # ResNet
    ("What problem does residual learning solve?",                     "degradation",       "resnet.pdf"),
    ("How deep is the deepest ResNet evaluated?",                      "152",               "resnet.pdf"),
    # ViT
    ("How does ViT split images into patches?",                        "patch",             "vit.pdf"),
    ("What large dataset does ViT need to outperform CNNs?",           "JFT",               "vit.pdf"),
    # CLIP
    ("How does CLIP train on image-text pairs?",                       "contrastive",       "clip.pdf"),
    ("How does CLIP do zero-shot classification?",                     "zero-shot",         "clip.pdf"),
    # GAN
    ("How does a GAN train generator and discriminator?",              "adversarial",       "gan.pdf"),
    ("What game-theoretic objective does GAN optimize?",               "minimax",           "gan.pdf"),
    # DDPM
    ("What is the forward diffusion process in DDPM?",                 "Markov",            "ddpm.pdf"),
    ("What architecture does DDPM use for denoising?",                 "U-Net",             "ddpm.pdf"),
    # GPT-3
    ("How many parameters does GPT-3 have?",                           "175",               "gpt3.pdf"),
    ("What is in-context learning in GPT-3?",                          "few-shot",          "gpt3.pdf"),
    # InstructGPT
    ("How is InstructGPT trained with human feedback?",                "reinforcement",     "instruct_gpt.pdf"),
    ("What is the reward model in InstructGPT?",                       "reward",            "instruct_gpt.pdf"),
    # LoRA
    ("How does LoRA reduce trainable parameters?",                     "low-rank",          "lora.pdf"),
    ("Does LoRA add inference latency?",                               "latency",           "lora.pdf"),
    # T5
    ("What input/output format does T5 use?",                          "text-to-text",      "t5.pdf"),
    ("What pre-training objective does T5 use?",                       "denoising",         "t5.pdf"),
    # SciBERT
    ("What corpus was SciBERT pre-trained on?",                        "Semantic Scholar",  "scibert.pdf"),
    ("How does SciBERT differ from BERT in vocabulary?",               "vocabulary",        "scibert.pdf"),
    # Word2Vec
    ("What is the skip-gram model in Word2Vec?",                       "skip-gram",         "word2vec.pdf"),
    ("What linguistic regularities does Word2Vec capture?",            "king",              "word2vec.pdf"),
]

# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 75)
print(f"EXTRACTIVE QA TEST — {len(QUESTIONS)} questions across 15 papers")
print("=" * 75)

correct = 0
t0 = time.time()
for i, (q, keyword, expected_pdf) in enumerate(QUESTIONS, 1):
    results = hybrid_search(q)
    answer  = extract_answer(q, results)

    # check keyword in answer AND answer came from correct paper
    hit = keyword.lower() in answer.lower() and expected_pdf in answer
    correct += hit
    mark = "✓" if hit else "✗"
    print(f"\nQ{i:02d} [{mark}] [{expected_pdf.replace('.pdf','')}] {q}")
    print(answer)

elapsed = time.time() - t0
print("\n" + "=" * 75)
print(f"Score : {correct}/{len(QUESTIONS)} ({100*correct//len(QUESTIONS)}%)")
print(f"Time  : {elapsed:.1f}s total  |  {elapsed/len(QUESTIONS)*1000:.0f}ms avg per query")
print("=" * 75)
