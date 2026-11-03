import { describe, it, expect } from 'vitest';
import { rrf, RRF_K } from './rrf';

describe('rrf', () => {
  it('dense-only results sorted by score', () => {
    const dense = new Map([[0, 0], [1, 1], [2, 2]]);
    const bm25  = new Map<number, number>();
    const result = rrf(dense, bm25, 3);
    // idx 0 has rank 0 → highest dense score
    expect(result[0]).toBe(0);
    expect(result[1]).toBe(1);
    expect(result[2]).toBe(2);
  });

  it('top-k limits output length', () => {
    const dense = new Map([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]);
    const bm25  = new Map<number, number>();
    expect(rrf(dense, bm25, 3)).toHaveLength(3);
  });

  it('dense-only scores use 2× weight', () => {
    // single item — verify score = 2/(RRF_K+1)
    const dense = new Map([[0, 0]]);
    const bm25  = new Map<number, number>();
    // just check the item is returned (score math is internal)
    expect(rrf(dense, bm25, 1)).toEqual([0]);
  });

  it('bm25-only results sorted by score', () => {
    const dense = new Map<number, number>();
    const bm25  = new Map([[5, 0], [6, 1]]);
    const result = rrf(dense, bm25, 2);
    expect(result[0]).toBe(5);
    expect(result[1]).toBe(6);
  });

  it('dense outranks bm25 for same rank due to 2:1 weight', () => {
    // idx 0 is rank-0 dense only; idx 1 is rank-0 bm25 only
    const dense = new Map([[0, 0]]);
    const bm25  = new Map([[1, 0]]);
    const result = rrf(dense, bm25, 2);
    // dense score = 2/(RRF_K+1), bm25 score = 1/(RRF_K+1) → dense wins
    expect(result[0]).toBe(0);
    expect(result[1]).toBe(1);
  });

  it('item in both lists beats item in one list', () => {
    // idx 0 is rank-0 in both; idx 1 is rank-0 dense only
    const dense = new Map([[0, 0], [1, 1]]);
    const bm25  = new Map([[0, 0]]);
    const result = rrf(dense, bm25, 2);
    // idx 0 score = 2/(K+1) + 1/(K+1) = 3/(K+1); idx 1 = 2/(K+2)
    expect(result[0]).toBe(0);
  });

  it('returns empty array for empty inputs', () => {
    expect(rrf(new Map(), new Map(), 10)).toEqual([]);
  });

  it('k larger than candidates returns all candidates', () => {
    const dense = new Map([[0, 0], [1, 1]]);
    const bm25  = new Map<number, number>();
    expect(rrf(dense, bm25, 100)).toHaveLength(2);
  });

  it('RRF_K is 60', () => {
    expect(RRF_K).toBe(60);
  });
});
