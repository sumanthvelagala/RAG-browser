/**
 * Reciprocal Rank Fusion (RRF) — 2:1 dense:BM25 weighting.
 * Extracted from main.ts so it can be unit-tested independently.
 */
export const RRF_K = 60;

export function rrf(
  denseRank: Map<number, number>,
  bm25Rank:  Map<number, number>,
  k:         number,
): number[] {
  const fused = new Map<number, number>();
  for (const [idx, rank] of denseRank) {
    fused.set(idx, (fused.get(idx) ?? 0) + 2.0 / (RRF_K + rank + 1));
  }
  for (const [idx, rank] of bm25Rank) {
    fused.set(idx, (fused.get(idx) ?? 0) + 1.0 / (RRF_K + rank + 1));
  }
  return [...fused.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, k)
    .map(([idx]) => idx);
}
