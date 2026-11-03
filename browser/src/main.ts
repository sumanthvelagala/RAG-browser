import './style.css';
import { loadPDF, renderPage, highlightChunk } from './pdfRenderer';
import type { PDFDocumentProxy } from './pdfRenderer';
import { chunkWords, type Chunk } from './chunker';
import { create, insertMultiple, search } from '@orama/orama';

// ── types ────────────────────────────────────────────────────────────────────
interface StoredChunk extends Chunk {
  id:    string;
  pdf:   string;
}

interface PdfEntry {
  name:       string;
  doc:        PDFDocumentProxy;
  blobUrl:    string;
  chunks:     StoredChunk[];
  embeddings: number[][];
  colEl:      HTMLElement;
  pagesEl:    HTMLElement;
  pageEls:    HTMLElement[];  // index = pageNum - 1
}

// ── state ────────────────────────────────────────────────────────────────────
const pdfs: Map<string, PdfEntry> = new Map();
const pendingPdfs = new Set<string>();   // in-flight ingestions
let totalPdfs  = 0;
let readyPdfs  = 0;
let totalPages = 0;
const PAGE_CAP = 500;

let countHideTimer: ReturnType<typeof setTimeout>;

function updatePdfCount() {
  clearTimeout(countHideTimer);
  if (totalPdfs === 0) { pdfCount.setAttribute('hidden', ''); return; }
  pdfCount.removeAttribute('hidden');
  if (readyPdfs < totalPdfs) {
    pdfCountNumber.textContent = `${readyPdfs} / ${totalPdfs}`;
    pdfCountLabel.textContent  = 'PDFs indexed';
    library.removeAttribute('data-indexed');
  } else {
    pdfCountNumber.textContent = String(totalPdfs);
    pdfCountLabel.textContent  = `PDF${totalPdfs !== 1 ? 's' : ''} ready`;
    library.setAttribute('data-indexed', 'true');
    countHideTimer = setTimeout(() => pdfCount.setAttribute('hidden', ''), 2000);
  }
}

// ── DOM refs ─────────────────────────────────────────────────────────────────
const library          = document.getElementById('library')!;
const emptyState       = document.getElementById('empty-state')!;
const searchInput      = document.getElementById('search-input') as HTMLInputElement;
const uploadBtn        = document.getElementById('upload-btn')!;
const fileInput        = document.getElementById('file-input') as HTMLInputElement;
const resultCount      = document.getElementById('results-count') as HTMLSelectElement;
const searchHint       = document.getElementById('search-hint')!;
const askBtn           = document.getElementById('ask-btn')!;
const pdfCount        = document.getElementById('pdf-count')!;
const pdfCountNumber  = document.getElementById('pdf-count-number')!;
const pdfCountLabel   = document.getElementById('pdf-count-label')!;
const aboutBtn         = document.getElementById('about-btn')!;
const modalBackdrop    = document.getElementById('modal-backdrop')!;
const modalClose       = document.getElementById('modal-close')!;
const pdfViewerBackdrop = document.getElementById('pdf-viewer-backdrop')!;
const pdfViewerClose   = document.getElementById('pdf-viewer-close')!;
const pdfViewerFrame   = document.getElementById('pdf-viewer-frame') as HTMLIFrameElement;

// ── embed worker pool — capped at min(hardwareConcurrency, 4) ─────────────────
const POOL_SIZE = Math.min(navigator.hardwareConcurrency || 2, 4);

interface PoolWorker {
  worker:    Worker;
  callbacks: Map<string, (e: number[][]) => void>;
  busy:      number;  // active batch count
}

const pool: PoolWorker[] = Array.from({ length: POOL_SIZE }, () => {
  const pw: PoolWorker = {
    worker:    new Worker(new URL('./embedder.worker.ts', import.meta.url), { type: 'module' }),
    callbacks: new Map(),
    busy:      0,
  };
  pw.worker.onmessage = (e: MessageEvent) => {
    const { type, id, embeddings, error, progress } = e.data;
    if (type === 'model_loading') setHint('Downloading embedding model (~22 MB)…');
    if (type === 'model_progress') setHint(`Downloading model… ${Math.round(progress ?? 0)}%`);
    if (type === 'model_ready')    setHint('');
    if (type === 'embed_result') {
      pw.busy--;
      pw.callbacks.get(id)?.(embeddings);
      pw.callbacks.delete(id);
    }
    if (type === 'embed_error') {
      pw.busy--;
      console.error('Embed error:', error);
      pw.callbacks.delete(id);
    }
  };
  return pw;
});

function embedTexts(texts: string[]): Promise<number[][]> {
  const pw = pool.reduce((a, b) => a.busy <= b.busy ? a : b);
  return new Promise(resolve => {
    const id = crypto.randomUUID();
    pw.callbacks.set(id, resolve);
    pw.busy++;
    pw.worker.postMessage({ id, texts });
  });
}

// ── PDF ingestion ─────────────────────────────────────────────────────────────
async function ingestPDF(file: File) {
  if (pdfs.has(file.name) || pendingPdfs.has(file.name)) return;
  pendingPdfs.add(file.name);

  totalPdfs++;
  updatePdfCount();
  emptyState.style.display = 'none';
  const col = buildColumn(file.name);
  library.appendChild(col.colEl);
  observeColumn(col.colEl);
  updateLibraryCentering();

  const buf     = await file.arrayBuffer();
  const blobUrl = URL.createObjectURL(new Blob([buf], { type: 'application/pdf' }));
  const doc     = await loadPDF(buf);

  // Pinch on mobile → open native PDF viewer (has proper pinch-zoom)
  col.pagesEl.addEventListener('touchstart', e => {
    if (e.touches.length === 2) {
      e.preventDefault();
      openPdfViewer(blobUrl, 1);
    }
  }, { passive: false });
  const numPages = doc.numPages;

  if (totalPages + numPages > PAGE_CAP) {
    const remaining = PAGE_CAP - totalPages;
    setHint(remaining > 0
      ? `"${file.name}" has ${numPages} pages but only ${remaining} remain in the ${PAGE_CAP}-page limit.`
      : `Page limit of ${PAGE_CAP} reached — remove a PDF to add more.`);
    URL.revokeObjectURL(blobUrl);
    library.removeChild(col.colEl);
    totalPdfs--;
    updatePdfCount();
    pendingPdfs.delete(file.name);
    return;
  }
  totalPages += numPages;

  // render all pages as thumbnails
  const pageEls: HTMLElement[] = [];
  for (let p = 1; p <= numPages; p++) {
    const card = buildLoadingCard(p);
    col.pagesEl.appendChild(card);
    pageEls.push(card);

    renderPage(doc, p).then(canvas => {
      card.innerHTML = '';
      card.appendChild(canvas);
      const label = document.createElement('div');
      label.className = 'page-num';
      label.textContent = String(p);
      card.appendChild(label);
      card.classList.remove('loading');
    });

    card.addEventListener('click', () => {
      if (card.classList.contains('collapsed')) {
        card.classList.remove('collapsed');
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        openPdfViewer(blobUrl, p);
      }
    });
  }

  // extract text and chunk per page
  const allWords: string[] = [];
  const boundaries: Array<[number, number]> = [];

  for (let p = 1; p <= numPages; p++) {
    const page    = await doc.getPage(p);
    const content = await page.getTextContent();
    const text    = (content.items as Array<{ str: string }>).map(i => i.str).join(' ');
    const words   = text.trim().split(/\s+/).filter(Boolean);
    if (words.length) {
      boundaries.push([allWords.length, p]);
      allWords.push(...words);
    }
  }

  const rawChunks = chunkWords(allWords, boundaries);
  const chunks: StoredChunk[] = rawChunks.map((c, i) => ({
    ...c,
    id:  `${file.name}::${i}`,
    pdf: file.name,
  }));

  col.barEl.style.width = '5%';

  const BATCH = 32;
  const allEmbeddings: number[][] = [];
  for (let i = 0; i < chunks.length; i += BATCH) {
    const batch = chunks.slice(i, i + BATCH).map(c => c.text);
    const embs  = await embedTexts(batch);
    allEmbeddings.push(...embs);
    const pct = Math.round(((i + batch.length) / chunks.length) * 100);
    col.barEl.style.width = `${pct}%`;
  }

  col.barEl.style.width = '100%';
  setTimeout(() => { (col.barEl.parentElement as HTMLElement).style.display = 'none'; }, 600);
  readyPdfs++;
  updatePdfCount();

  pendingPdfs.delete(file.name);
  pdfs.set(file.name, {
    name: file.name,
    doc,
    blobUrl,
    chunks,
    embeddings: allEmbeddings,
    colEl: col.colEl,
    pagesEl: col.pagesEl,
    pageEls,
  });

}

// ── search ────────────────────────────────────────────────────────────────────

searchInput.addEventListener('input', () => {
  if (!searchInput.value.trim()) clearResults();
});

searchInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') runSearch(searchInput.value.trim());
});

askBtn.addEventListener('click', () => {
  runSearch(searchInput.value.trim());
});

async function runSearch(query: string) {
  if (!query) return;
  if (pdfs.size === 0) {
    setHint(pendingPdfs.size > 0 ? 'Still indexing — try again in a moment.' : 'Add a PDF first.');
    return;
  }
  const k = parseInt(resultCount.value, 10);

  setHint('Searching…');

  const [queryEmb] = await embedTexts([query]);

  // gather all chunks + embeddings across pdfs
  const allChunks: StoredChunk[]  = [];
  const allEmbs:   number[][]     = [];
  for (const entry of pdfs.values()) {
    allChunks.push(...entry.chunks);
    allEmbs.push(...entry.embeddings);
  }

  if (allChunks.length === 0) { setHint('Add a PDF first.'); return; }

  // BM25 via Orama (rebuilt each search — fast for 100s of docs)
  const db = await create({ schema: { text: 'string', idx: 'number' } });
  await insertMultiple(db, allChunks.map((c, i) => ({ text: c.text, idx: i })));
  const bm25Res = await search(db, { term: query, properties: ['text'], limit: Math.max(50, 3 * k), threshold: 1 });
  const bm25Rank = new Map<number, number>();
  bm25Res.hits.forEach((h, rank) => bm25Rank.set((h.document as { idx: number }).idx, rank));

  // dense cosine scores
  const sims = allEmbs.map(emb => dot(emb, queryEmb));
  const pool  = Math.max(50, 3 * k);
  const denseRanked = argsort(sims, true).slice(0, pool);
  const denseRank   = new Map<number, number>();
  denseRanked.forEach((idx, rank) => denseRank.set(idx, rank));

  // RRF fusion (2:1 dense:BM25)
  const RRF_K = 60;
  const fused = new Map<number, number>();
  for (const [idx, rank] of denseRank) {
    fused.set(idx, (fused.get(idx) ?? 0) + 2.0 / (RRF_K + rank + 1));
  }
  for (const [idx, rank] of bm25Rank) {
    fused.set(idx, (fused.get(idx) ?? 0) + 1.0 / (RRF_K + rank + 1));
  }

  const topK = [...fused.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, k)
    .map(([idx]) => allChunks[idx]);

  applyResults(topK);
  setHint(`${topK.length} result${topK.length !== 1 ? 's' : ''} · click any page to open`);
}

function applyResults(results: StoredChunk[]) {
  clearResults();
  library.classList.add('has-results');

  // group results by "pdf::page"
  const matchedKeys = new Map<string, StoredChunk[]>();
  const matchedPdfs = new Set<string>();
  for (const c of results) {
    const key = `${c.pdf}::${c.page}`;
    if (!matchedKeys.has(key)) matchedKeys.set(key, []);
    matchedKeys.get(key)!.push(c);
    matchedPdfs.add(c.pdf);
  }

  for (const [name, entry] of pdfs) {
    if (!matchedPdfs.has(name)) {
      entry.colEl.style.display = 'none';
      continue;
    }

    let firstMatch: HTMLElement | null = null;

    for (const [i, card] of entry.pageEls.entries()) {
      const pageNum  = i + 1;
      const chunks   = matchedKeys.get(`${name}::${pageNum}`);
      if (chunks) {
        card.classList.add('matched');
        if (!firstMatch) firstMatch = card;
        const canvas = card.querySelector('canvas') as HTMLCanvasElement | null;
        if (canvas) {
          for (const chunk of chunks) highlightChunk(canvas, entry.doc, pageNum, chunk.text);
        }
      } else {
        card.classList.add('collapsed');
      }
    }

    firstMatch?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function clearResults() {
  library.classList.remove('has-results');
  for (const entry of pdfs.values()) {
    entry.colEl.style.display = '';
    for (const [i, card] of entry.pageEls.entries()) {
      card.classList.remove('matched', 'collapsed');
      const canvas = card.querySelector('canvas') as HTMLCanvasElement | null;
      if (canvas) {
        renderPage(entry.doc, i + 1).then(fresh => {
          canvas.getContext('2d')!.clearRect(0, 0, canvas.width, canvas.height);
          canvas.getContext('2d')!.drawImage(fresh, 0, 0);
        });
      }
    }
  }
}

// ── DOM builders ──────────────────────────────────────────────────────────────
function buildColumn(name: string) {
  const colEl = document.createElement('div');
  colEl.className = 'pdf-col';

  const headerEl = document.createElement('div');
  headerEl.className = 'pdf-col-header';
  headerEl.innerHTML = `<span>Loading…</span>`;
  colEl.appendChild(headerEl);

  const barWrap = document.createElement('div');
  barWrap.className = 'index-bar';
  const barEl = document.createElement('div');
  barEl.className = 'index-bar-fill';
  barEl.style.width = '0%';
  barWrap.appendChild(barEl);
  colEl.appendChild(barWrap);

  const pagesEl = document.createElement('div');
  pagesEl.className = 'pdf-col-pages';
  colEl.appendChild(pagesEl);

  const footerEl = document.createElement('div');
  footerEl.className = 'pdf-col-footer';
  footerEl.textContent = name;
  colEl.appendChild(footerEl);

  return { colEl, headerEl, barEl, pagesEl };
}

function buildLoadingCard(page: number): HTMLElement {
  const card = document.createElement('div');
  card.className = 'page-card loading';
  card.dataset['page'] = String(page);
  const spinner = document.createElement('div');
  spinner.className = 'spinner';
  card.appendChild(spinner);
  return card;
}

// ── helpers ───────────────────────────────────────────────────────────────────
function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function argsort(arr: number[], desc = false): number[] {
  return Array.from({ length: arr.length }, (_, i) => i)
    .sort((a, b) => desc ? arr[b] - arr[a] : arr[a] - arr[b]);
}

function setHint(msg: string) {
  searchHint.textContent = msg;
}

// ── upload wiring ─────────────────────────────────────────────────────────────
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  const files = Array.from(fileInput.files ?? []);
  fileInput.value = '';
  files.forEach(f => ingestPDF(f));
});

// ── pdf viewer ────────────────────────────────────────────────────────────────
function openPdfViewer(blobUrl: string, page: number) {
  pdfViewerFrame.src = `${blobUrl}#page=${page}`;
  pdfViewerBackdrop.removeAttribute('hidden');
}

function closePdfViewer() {
  pdfViewerBackdrop.setAttribute('hidden', '');
  pdfViewerFrame.src = '';
}

pdfViewerClose.addEventListener('click', closePdfViewer);
pdfViewerBackdrop.addEventListener('click', e => {
  if (e.target === pdfViewerBackdrop) closePdfViewer();
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closePdfViewer();
});

// ── center-column pop ─────────────────────────────────────────────────────────
const colObserver = new IntersectionObserver(
  entries => {
    for (const e of entries) {
      e.target.classList.toggle('col-active', e.isIntersecting);
    }
  },
  { root: library, rootMargin: '0px -34% 0px -34%', threshold: 0 },
);

function observeColumn(colEl: HTMLElement) {
  colObserver.observe(colEl);
}

function updateLibraryCentering() { /* spacers handle centering */ }

// ── about modal ───────────────────────────────────────────────────────────────
aboutBtn.addEventListener('click', () => modalBackdrop.removeAttribute('hidden'));
modalClose.addEventListener('click', () => modalBackdrop.setAttribute('hidden', ''));
modalBackdrop.addEventListener('click', e => {
  if (e.target === modalBackdrop) modalBackdrop.setAttribute('hidden', '');
});

// ── init ──────────────────────────────────────────────────────────────────────
(function addLibrarySpacers() {
  const mk = () => { const d = document.createElement('div'); d.className = 'library-spacer'; return d; };
  library.insertBefore(mk(), library.firstChild);
  library.appendChild(mk());
}());
