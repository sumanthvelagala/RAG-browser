import * as pdfjsLib from 'pdfjs-dist';
import type { PDFDocumentProxy, PDFPageProxy } from 'pdfjs-dist';

// Vite resolves this to the correct URL at build time
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

export type { PDFDocumentProxy, PDFPageProxy };

// Render at 2× base scale × device pixel ratio for crisp text at any column width
const BASE_SCALE = 2.0;
const PAGE_SCALE = BASE_SCALE * (typeof devicePixelRatio !== 'undefined' ? Math.min(devicePixelRatio, 2) : 1);

export async function loadPDF(arrayBuffer: ArrayBuffer): Promise<PDFDocumentProxy> {
  return pdfjsLib.getDocument({ data: arrayBuffer }).promise;
}

/** Render a single page into a <canvas> and return it. */
export async function renderPage(
  pdfDoc: PDFDocumentProxy,
  pageNum: number,
): Promise<HTMLCanvasElement> {
  const page   = await pdfDoc.getPage(pageNum);
  const vp     = page.getViewport({ scale: PAGE_SCALE });
  const canvas = document.createElement('canvas');
  canvas.width  = Math.floor(vp.width);
  canvas.height = Math.floor(vp.height);
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  await page.render({ canvasContext: ctx, viewport: vp, canvas }).promise;
  return canvas;
}

type TItem = { str: string; transform: number[]; width: number; height: number };

/** Strip punctuation + lowercase for fuzzy word matching */
const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '');

/**
 * Sequence-match the retrieved chunk in the page text layer and highlight it.
 * Uses the first 8 words as an anchor to locate the chunk, then highlights
 * all text items that cover the chunk's word range.
 */
export async function highlightChunk(
  canvas: HTMLCanvasElement,
  pdfDoc: PDFDocumentProxy,
  pageNum: number,
  chunkText: string,
): Promise<void> {
  const page    = await pdfDoc.getPage(pageNum);
  const vp      = page.getViewport({ scale: PAGE_SCALE });
  const content = await page.getTextContent();
  const ctx     = canvas.getContext('2d')!;
  const items   = content.items as TItem[];

  // build flat (word → item) list from page
  const pageWords: Array<{ w: string; item: TItem }> = [];
  for (const item of items) {
    for (const w of item.str.split(/\s+/).filter(Boolean)) {
      pageWords.push({ w, item });
    }
  }

  // find chunk start using first 8 non-trivial words as anchor
  const chunkWords = chunkText.split(/\s+/).filter(Boolean);
  const anchor     = chunkWords.slice(0, 8).map(norm).filter(w => w.length > 1);
  let   matchStart = -1;

  outer: for (let i = 0; i <= pageWords.length - anchor.length; i++) {
    for (let j = 0; j < anchor.length; j++) {
      if (norm(pageWords[i + j]?.w ?? '') !== anchor[j]) continue outer;
    }
    matchStart = i;
    break;
  }

  if (matchStart < 0) return;

  // highlight all items covering the chunk word range
  const end         = Math.min(matchStart + chunkWords.length, pageWords.length);
  const regionItems = new Set(pageWords.slice(matchStart, end).map(x => x.item));

  ctx.save();
  ctx.fillStyle = 'rgba(254, 240, 138, 0.38)';
  for (const item of regionItems) {
    const [,,,, tx, ty] = item.transform;
    const pt = vp.convertToViewportPoint(tx, ty);
    ctx.fillRect(pt[0], pt[1] - item.height * PAGE_SCALE, item.width * PAGE_SCALE, item.height * PAGE_SCALE * 1.15);
  }
  ctx.restore();
}
