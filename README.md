# PDF Library Search — In-Browser, Zero-Install

Search your own PDFs privately in the browser. No account, no server, no upload.
Text extraction, embedding, BM25, and extractive QA all run on your device.
Results jump to the exact page and highlight the matching passage.

> **Note:** update the GitHub "About" description to match — the repo description
> should say "in-browser hybrid retrieval (dense + BM25) with extractive QA",
> not "LLM generates answers". This project uses no generative LLM.

## How it works

1. Drop PDFs into the browser. PDF.js extracts text; no bytes leave your device.
2. Chunks are embedded with `all-MiniLM-L6-v2` (quantized, ~23 MB) running in a
   Web Worker via Transformers.js.
3. A JS BM25 index is built in parallel. Both indexes are saved to IndexedDB so
   re-indexing is skipped on reload.
4. On query: dense retrieval + BM25 top-50 each, fused with RRF (2:1 dense:BM25),
   extractive QA returns the best verbatim sentences with zero hallucination risk.
5. Click a result — PDF.js renders the page and the matching passage is highlighted.

Everything runs offline after the first load (model download ~23 MB).

## Privacy

- PDFs are never uploaded. All processing is local.
- The embedding model is fetched once from Hugging Face and cached by the browser.
- No analytics touch document content. At most, privacy-friendly page visit counts.
- Turn off Wi-Fi after the page loads — it still works.

## Eval results (Python reference, 60 questions, 15 papers, 6 domains)

Measured locally (2 runs, stable). Per-page extraction (Phase 1a) improved @10
by 1.7 points vs the earlier full-document extraction, by preventing text from
bleeding across page boundaries.

| Metric | Hybrid RRF (2:1) | BM25 baseline |
|---|---|---|
| Recall@5  | **80.0% (48/60)** | 58.3% (35/60) |
| Recall@10 | **91.7% (55/60)** | 68.3% (41/60) |
| Recall@20 | **93.3% (56/60)** | 81.7% (49/60) |
| Avg latency @10 | **25ms** | <1ms |

Browser latency will be measured and reported here once Phase 4 is complete.

## Accuracy journey (Python reference, Recall@10 unless noted)

| Step | Change | Recall@10 |
|---|---|---|
| Baseline | SciBERT CLS embeddings | 0% |
| Swap model | all-MiniLM-L6-v2 | 55% |
| Better model | all-mpnet-base-v2 | 65% |
| Smaller chunks | 200 → 128 tokens | 70% |
| Hybrid RRF (1:1) | dense + BM25 fusion | 85% |
| Weighted RRF (2:1) | dense 2x weight | 90% |
| Universal chunker | remove title heuristic | 90.0% |
| Per-page extraction | chunk within pages, no cross-page bleed | **91.7%** |

## Models & approach

- **Browser embeddings**: `all-MiniLM-L6-v2` quantized int8 (~23 MB, fits
  Cloudflare Pages' 25 MiB per-file limit). `all-mpnet-base-v2` (~420 MB) would
  improve accuracy but requires hosting on R2; tracked as a future option.
- **Answer generation**: Extractive QA — no LLM. Top-10 retrieved chunks are split
  into sentences, each scored as `cosine_sim(sentence, query) × chunk_rank_weight`.
  Top-5 are returned with semantic dedup (skip if sim > 0.90 to already-selected).
  Returns "Not found in the document." if no sentence clears a 0.50 threshold.
  Verbatim sentences only — zero hallucination risk.
- **Python reference**: `all-mpnet-base-v2` via sentence-transformers + ChromaDB.

**Why not a generative LLM?** Tested TinyLlama-1.1B on 15 diverse questions:
scored 5/15 (33%), including hallucinating inverted answers ("pets are allowed"
when context says no). Extractive QA scored 14/15 (93%) on the same eval with
zero factual inversions.

## Roadmap

| Phase | What | Status |
|---|---|---|
| 1 | Page-aware chunks + passage highlighting in Python app (PyMuPDF) | planned |
| 2 | TypeScript + Vite browser port — PDF.js, Transformers.js, JS BM25, IndexedDB, Cloudflare Pages | planned |
| 3 | Browser passage highlighting — click result to jump to page | planned |
| 4 | Headless Playwright eval against browser version; report Recall@10 + latency vs server | planned |
| 5 | Unit tests, CI, demo GIF, architecture diagram, changelog, tagged release | planned |

## Known limitations

- Scanned PDFs (images only) are not supported — OCR is out of scope for v1.
- Tables and equations are extracted as plain text and may be garbled.
- Two-column academic papers (e.g. most ML papers) have reading-order problems:
  PDF text extraction reads across columns, mixing left and right column text.
  This degrades chunk quality for those layouts.
- v1 cap: ~300 documents.

## Python reference implementation

The `app.py` Gradio app and associated pipeline remain the reference. Run locally:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Gradio app
python app.py

# Eval pipeline (requires papers/ directory)
uvicorn main:app --host 127.0.0.1 --port 8000
python ingest.py --fresh
python eval.py
```

## Python reference files

| File | Purpose |
|---|---|
| `app.py` | Gradio app — per-session upload + hybrid search + extractive QA |
| `front.py` | Streamlit app — local development |
| `main.py` | FastAPI backend — `/store` and `/query` endpoints |
| `ingest.py` | Batch ingest papers into persistent ChromaDB for eval |
| `eval.py` | Recall@10 evaluation: hybrid RRF vs BM25 baseline |
| `embedding.py` | all-mpnet-base-v2 encode wrapper |
| `chunking.py` | Universal sliding window chunker |
| `cleaners.py` | PDF text cleaning (hyphen breaks, page numbers, whitespace) |
