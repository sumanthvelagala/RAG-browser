/**
 * Sliding window chunker — mirrors chunking.py exactly.
 * Returns {text, page} pairs.
 */
export interface Chunk {
  text: string;
  page: number;
}

export function chunkWords(
  words: string[],
  pageBoundaries: Array<[number, number]>, // [startWordIdx, pageNum]
  tokenLimit = 128,
  overlap = 32,
): Chunk[] {
  if (words.length === 0) return [];

  const step = tokenLimit - overlap;
  const starts = pageBoundaries.map(([s]) => s);
  const pages  = pageBoundaries.map(([, p]) => p);
  const chunks: Chunk[] = [];

  for (let start = 0; start < words.length; start += step) {
    const slice = words.slice(start, start + tokenLimit);
    if (slice.length === 0) break;

    // last boundary with startWordIdx <= start
    let lo = 0, hi = starts.length - 1, pageIdx = 0;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (starts[mid] <= start) { pageIdx = mid; lo = mid + 1; }
      else hi = mid - 1;
    }

    chunks.push({ text: slice.join(' '), page: pages[pageIdx] ?? 1 });
  }

  return chunks;
}
