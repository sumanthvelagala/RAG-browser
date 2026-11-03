# PDF Search — In-Browser, Zero-Install

**Live:** https://pdf-search-79n.pages.dev

Search your own PDFs privately in the browser. No account, no server, no upload.
Text extraction, embedding, BM25, and hybrid retrieval all run on your device.
Results jump to the exact page and highlight the matching passage.

## How it works

1. Drop PDFs into the browser. PDF.js extracts text — no bytes leave your device.
2. Chunks are embedded with `all-MiniLM-L6-v2` (quantized, ~22 MB) running in a
   Web Worker via Transformers.js.
3. A BM25 index is built in parallel using Orama.
4. On query: dense retrieval + BM25 top-50 each, fused with RRF (2:1 dense:BM25),
   top-k results returned with zero hallucination risk.
5. Click a result — PDF opens in a native viewer at the exact page.

Everything runs offline after the first load (model download ~22 MB, cached by browser).

## Privacy

- PDFs are never uploaded. All processing is local.
- The embedding model is fetched once from Hugging Face CDN and cached by the browser.
- No analytics touch document content.
- Turn off Wi-Fi after the page loads — it still works.

## Limits

- **500 pages total** across all loaded PDFs. Mix of many small docs or one large one.
- Scanned PDFs (image-only) are not supported — no OCR.
- Two-column layouts (most ML papers) have reading-order issues in text extraction.
- Tables and equations extracted as plain text may be garbled.

## Eval results (Python reference, 60 questions, 15 papers, 6 domains)

| Metric | Hybrid RRF (2:1) | BM25 baseline |
|---|---|---|
| Recall@5  | **80.0% (48/60)** | 58.3% (35/60) |
| Recall@10 | **91.7% (55/60)** | 68.3% (41/60) |
| Recall@20 | **93.3% (56/60)** | 81.7% (49/60) |
| Avg latency @10 | **25ms** | <1ms |

## Models

- **Browser embeddings**: `all-MiniLM-L6-v2` quantized int8 (~22 MB). The ONNX
  runtime WASM (26 MB) exceeds Cloudflare Pages' 25 MiB per-file limit and is
  loaded at runtime from jsDelivr CDN.
- **Python reference**: `all-mpnet-base-v2` via sentence-transformers + ChromaDB.

## Browser app

```bash
cd browser
npm install
npm run dev      # http://localhost:5173
npm run build    # production build → dist/
```

### Browser source files

| File | Purpose |
|---|---|
| `src/main.ts` | App entry — PDF ingestion, search (RRF fusion), result rendering, UI wiring |
| `src/embedder.worker.ts` | Web Worker — runs `all-MiniLM-L6-v2` off the main thread via Transformers.js |
| `src/pdfRenderer.ts` | PDF.js wrapper — loads PDFs, renders pages to canvas at 2× DPI, highlights matched chunks |
| `src/chunker.ts` | Sliding window chunker (mirrors `chunking.py`) — returns `{text, page}` pairs |
| `src/style.css` | All UI styles |
| `index.html` | App shell — layout, modals, search bar |
| `vite.config.ts` | Vite build config — removes oversized ONNX asyncify WASM (26 MB) from bundle |
| `public/_headers` | Cloudflare Pages response headers — `COEP: credentialless`, `COOP: same-origin` for SharedArrayBuffer |

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
| `eval.py` | Recall@10 evaluation: hybrid RRF vs BM25 baseline |
| `embedding.py` | all-mpnet-base-v2 encode wrapper |
| `chunking.py` | Universal sliding window chunker |
| `cleaners.py` | PDF text cleaning (hyphen breaks, page numbers, whitespace) |
