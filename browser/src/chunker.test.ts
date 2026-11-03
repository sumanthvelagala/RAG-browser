import { describe, it, expect } from 'vitest';
import { chunkWords } from './chunker';

function makeWords(n: number): string[] {
  return Array.from({ length: n }, (_, i) => `w${i}`);
}

describe('chunkWords', () => {
  it('returns empty array for empty input', () => {
    expect(chunkWords([], [])).toEqual([]);
  });

  it('single chunk when words fit in one window', () => {
    const words = makeWords(50);
    const chunks = chunkWords(words, [[0, 1]]);
    expect(chunks).toHaveLength(1);
    expect(chunks[0].text).toBe(words.join(' '));
  });

  it('chunk text does not exceed token limit', () => {
    const words = makeWords(300);
    const chunks = chunkWords(words, [[0, 1]], 128, 32);
    for (const c of chunks) {
      expect(c.text.split(' ').length).toBeLessThanOrEqual(128);
    }
  });

  it('overlap: second chunk starts 96 words in (step = 128-32)', () => {
    const words = makeWords(300);
    const chunks = chunkWords(words, [[0, 1]], 128, 32);
    const second = chunks[1].text.split(' ');
    // second chunk starts at word index 96
    expect(second[0]).toBe('w96');
  });

  it('assigns correct page from pageBoundaries', () => {
    const words = makeWords(200);
    // page 1 starts at word 0, page 2 starts at word 100
    const chunks = chunkWords(words, [[0, 1], [100, 2]], 128, 32);
    // first chunk (starts at 0) → page 1
    expect(chunks[0].page).toBe(1);
    // chunk starting at 96 → still before boundary 100 → page 1
    expect(chunks[1].page).toBe(1);
    // chunk starting at 192 → after boundary 100 → page 2
    const lastChunk = chunks[chunks.length - 1];
    expect(lastChunk.page).toBe(2);
  });

  it('page boundary at exact chunk start', () => {
    const words = makeWords(300);
    // page 2 starts exactly at word 96 (second chunk start)
    const chunks = chunkWords(words, [[0, 1], [96, 2]], 128, 32);
    expect(chunks[1].page).toBe(2);
  });

  it('defaults to page 1 when pageBoundaries has one entry', () => {
    const words = makeWords(50);
    const chunks = chunkWords(words, [[0, 5]]);
    expect(chunks[0].page).toBe(5);
  });

  it('produces overlapping text between consecutive chunks', () => {
    const words = makeWords(200);
    const chunks = chunkWords(words, [[0, 1]], 128, 32);
    const first  = chunks[0].text.split(' ');
    const second = chunks[1].text.split(' ');
    // last 32 words of first == first 32 words of second
    expect(first.slice(-32)).toEqual(second.slice(0, 32));
  });
});
