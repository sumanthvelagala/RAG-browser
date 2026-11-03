# Changelog

All notable changes to this project will be documented here.

## [0.1.0] — 2026-11-03

### Added
- In-browser hybrid PDF search: dense embeddings (`all-MiniLM-L6-v2` q8 via Transformers.js) + BM25 (Orama), fused with RRF (2:1 dense:BM25)
- PDF.js text extraction and page rendering at 2× DPI with matched-passage highlighting
- Result-count selector (5 / 10 / 20, default 10); retrieval pool = max(50, 3×N)
- 500-page cap across all loaded PDFs
- Duplicate PDF rejection (race-condition safe via `pendingPdfs` set)
- Eval script (`eval.ts`): 70 questions over 15 papers — Recall@10 83.3% hybrid vs 41.7% BM25-only
- Vitest unit tests for chunker (overlap, page boundaries) and RRF fusion (2:1 weighting, top-k)
- GitHub Actions CI: type-check → unit tests → build → dist check
- Cloudflare Pages deployment at https://pdf-search-79n.pages.dev
- MIT license
