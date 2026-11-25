# In-Browser Semantic PDF Search

**Live:** https://pdf-search-79n.pages.dev

Search your own PDFs privately in the browser. No account, no server, no upload.
Text extraction, embedding, BM25, and hybrid retrieval all run on your device.
Results jump to the exact page and highlight the matching passage.

![Demo — upload PDFs, search with simple and hard semantic queries](demo.gif)

## How it works

1. Drop PDFs into the browser. PDF.js extracts text — no bytes leave your device.
2. Chunks are embedded with `all-MiniLM-L6-v2` (quantized, ~22 MB) running in a
   Web Worker via Transformers.js. The ONNX runtime loads from jsDelivr CDN on
   first use; the model weights load from Hugging Face CDN once and are cached
   by the browser. Documents themselves are never uploaded anywhere.
3. A BM25 index is built in parallel using Orama.
4. On query: dense retrieval + BM25 top-50 each, fused with RRF (2:1 dense:BM25),
   top-k results returned — verbatim passages from your documents, no generation.
5. Matched pages are rendered with PDF.js at 2× DPI and the matched passage is
   highlighted. Click any page to open the full PDF at that page.

After the first load the app works offline — the ONNX runtime (~26 MB) is cached by
the service worker and the model weights (~22 MB) are cached by Transformers.js.
The very first visit requires network to fetch both.

## Privacy

- PDFs are never uploaded. All processing is local.
- The embedding model is fetched once from Hugging Face CDN and cached by the browser.
- No analytics touch document content.
- Turn off Wi-Fi after the first load — it still works (ONNX runtime and app shell are cached by the service worker).

## Known limits

| Limit | Detail |
|---|---|
| **500 pages total** | Across all loaded PDFs combined |
| **No OCR** | Scanned (image-only) PDFs return no text |
| **Two-column layouts** | Reading order issues in text extraction (affects most ML papers) |
| **Tables and equations** | Extracted as plain text — may be garbled |

## Eval results

Both pipelines share the same 60 questions across 15 papers, 7 domains.
The browser eval (`eval.ts`) also runs 10 additional hard semantic queries (70 total).
Browser eval run with `browser/eval.ts` (tsx + Transformers.js); Python eval run with `eval.py`.

| Metric | Browser · `all-MiniLM-L6-v2` q8 | Python ref · `all-mpnet-base-v2` |
|---|---|---|
| Recall@5  | 68.3% (41/60) | **80.0% (48/60)** |
| Recall@10 | 83.3% (50/60) | **91.7% (55/60)** |
| Recall@20 | 91.7% (55/60) | **93.3% (56/60)** |

Gap at Recall@10: **−8.4 pp** — model size (384-dim q8 vs 768-dim fp32) and quantization.

**Hard semantic queries** (10 questions, vocabulary deliberately different from document text):

| Metric | Browser hybrid | Browser BM25 only† |
|---|---|---|
| Recall@10 | 50.0% (5/10) | 10.0% (1/10) |
| Recall@20 | 80.0% (8/10) | 10.0% (1/10) |

All 5 hybrid passes come from dense retrieval; BM25 alone finds only 1 of the 10.

† BM25 uses `threshold:1` (all query tokens must match) — same setting as the live app.
Python BM25 baseline uses soft scoring so numbers are not directly comparable.

## Models

- **Browser**: `all-MiniLM-L6-v2` quantized int8 (~22 MB). The ONNX runtime WASM
  (26 MB) exceeds Cloudflare Pages' 25 MiB per-file limit and loads from jsDelivr CDN.
- **Python reference**: `all-mpnet-base-v2` via sentence-transformers + ChromaDB.

## Browser app

```bash
cd browser
npm install
npm run dev      # http://localhost:5173
npm run build    # production build → dist/
npx tsx eval.ts  # run browser pipeline eval (70 questions)
```

### Browser source files

| File | Purpose |
|---|---|
| `src/main.ts` | App entry — PDF ingestion, search (RRF fusion), result rendering, UI wiring |
| `src/embedder.worker.ts` | Web Worker — runs `all-MiniLM-L6-v2` off the main thread via Transformers.js |
| `src/pdfRenderer.ts` | PDF.js wrapper — loads PDFs, renders pages to canvas at 2× DPI, highlights matched chunks |
| `src/chunker.ts` | Sliding window chunker (mirrors `chunking.py`) — returns `{text, page}` pairs |
| `src/rrf.ts` | Reciprocal Rank Fusion — combines dense + BM25 rankings with 2:1 weighting, returns top-k indices |
| `src/style.css` | All UI styles |
| `index.html` | App shell — layout, modals, search bar |
| `public/sw.js` | Service worker — caches the ONNX runtime (cache-first) and app shell (stale-while-revalidate) for offline use |
| `vite.config.ts` | Vite build config — removes oversized ONNX asyncify WASM (26 MB) from bundle |
| `public/_headers` | Cloudflare Pages response headers — `COEP: credentialless`, `COOP: same-origin` for SharedArrayBuffer |
| `eval.ts` | Browser pipeline eval — 60 shared + 10 hard semantic questions, Transformers.js + Orama + RRF |

## Python reference implementation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python app.py                                    # Gradio app
uvicorn main:app --host 127.0.0.1 --port 8000   # FastAPI backend
python ingest.py --fresh                         # batch ingest
python eval.py                                   # Recall@10 eval
```

## Python reference files

| File | Purpose |
|---|---|
| `app.py` | Gradio app — per-session upload + hybrid search |
| `front.py` | Streamlit app — local development |
| `main.py` | FastAPI backend — `/store` and `/query` endpoints |
| `ingest.py` | Batch ingest papers into persistent ChromaDB for eval |
| `eval.py` | Recall@5/10/20 evaluation: hybrid RRF vs BM25 baseline |
| `embedding.py` | all-mpnet-base-v2 encode wrapper |
| `chunking.py` | Universal sliding window chunker |
| `cleaners.py` | PDF text cleaning (hyphen breaks, page numbers, whitespace) |
