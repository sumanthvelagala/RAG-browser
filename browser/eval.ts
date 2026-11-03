/**
 * Browser pipeline eval — mirrors eval.py but uses the actual browser stack:
 *   chunker.ts  ·  @orama/orama BM25  ·  all-MiniLM-L6-v2 q8 via Transformers.js
 *
 * Run:  cd browser && npx tsx eval.ts
 */

import { readFile } from 'fs/promises';
import { join } from 'path';
import { pipeline } from '@huggingface/transformers';
import { create, insertMultiple, search } from '@orama/orama';
import { chunkWords } from './src/chunker.ts';

const PAPERS_DIR = join(import.meta.dirname, '../papers');
const EVAL_KS    = [5, 10, 20] as const;
const RRF_K      = 60;
const BATCH_SIZE = 32;

// ── 60 test cases (same as eval.py) ──────────────────────────────────────────
// [question, keyword, pdfId]
const CASES: [string, string, string][] = [
  // BERT
  ['How is BERT pre-trained?',                              'masked',            'bert.pdf'],
  ['What tasks does BERT fine-tune on?',                   'fine-tun',          'bert.pdf'],
  ['What is the BERT model architecture?',                 'Transformer',       'bert.pdf'],
  ['What datasets are used to pre-train BERT?',            'BooksCorpus',       'bert.pdf'],
  // Attention Is All You Need
  ['What is the Transformer architecture?',                'encoder',           'attention_is_all_you_need.pdf'],
  ['What attention mechanism does the Transformer use?',   'multi-head',        'attention_is_all_you_need.pdf'],
  ['What optimizer was used to train the Transformer?',    'Adam',              'attention_is_all_you_need.pdf'],
  ['What is positional encoding in the Transformer?',      'positional',        'attention_is_all_you_need.pdf'],
  // RAG Lewis
  ['How does RAG combine retrieval with generation?',      'retrieval',         'rag_lewis.pdf'],
  ['What retriever does RAG use?',                         'DPR',               'rag_lewis.pdf'],
  ['What datasets are used to evaluate RAG?',              'Natural Questions', 'rag_lewis.pdf'],
  ['What is the generator model used in RAG?',             'BART',              'rag_lewis.pdf'],
  // SciBERT
  ['What corpus was SciBERT trained on?',                  'Semantic Scholar',  'scibert.pdf'],
  ['How does SciBERT compare to BERT on scientific tasks?','F1',                'scibert.pdf'],
  ['What vocabulary does SciBERT use?',                    'vocabulary',        'scibert.pdf'],
  ['What scientific domains does SciBERT cover?',          'biomedical',        'scibert.pdf'],
  // DPR
  ['What is dense passage retrieval?',                     'passage',           'dpr.pdf'],
  ['How does DPR differ from BM25?',                       'BM25',              'dpr.pdf'],
  ['What encoder architecture does DPR use?',              'BERT',              'dpr.pdf'],
  ['What open domain QA datasets evaluate DPR?',           'Natural Questions', 'dpr.pdf'],
  // ResNet
  ['What problem does residual learning solve in deep networks?', 'degradation', 'resnet.pdf'],
  ['How does ResNet use skip connections?',                'shortcut',          'resnet.pdf'],
  ['What dataset does ResNet evaluate on?',                'ImageNet',          'resnet.pdf'],
  ['How deep is the deepest ResNet model evaluated?',      '152',               'resnet.pdf'],
  // CLIP
  ['How does CLIP train on image and text pairs?',         'contrastive',       'clip.pdf'],
  ['What dataset is CLIP pre-trained on?',                 'WIT',               'clip.pdf'],
  ['How does CLIP perform zero-shot classification?',      'zero-shot',         'clip.pdf'],
  ['What image encoder does CLIP use?',                    'ViT',               'clip.pdf'],
  // T5
  ['What input and output format does T5 use?',            'text-to-text',      't5.pdf'],
  ['What pre-training dataset does T5 use?',               'C4',                't5.pdf'],
  ['What pre-training objective does T5 use?',             'denoising',         't5.pdf'],
  ['What is the largest T5 model size evaluated?',         '11',                't5.pdf'],
  // GPT-3
  ['What is in-context learning in GPT-3?',                'in-context',        'gpt3.pdf'],
  ['How many parameters does GPT-3 have?',                 '175',               'gpt3.pdf'],
  ['What training data does GPT-3 use?',                   'Common Crawl',      'gpt3.pdf'],
  ['How does GPT-3 perform few-shot learning?',            'few-shot',          'gpt3.pdf'],
  // InstructGPT
  ['How is InstructGPT trained with human feedback?',      'reinforcement',     'instruct_gpt.pdf'],
  ['What is the reward model in InstructGPT?',             'reward',            'instruct_gpt.pdf'],
  ['What labelers do for InstructGPT training?',           'labeler',           'instruct_gpt.pdf'],
  ['How does InstructGPT align language models?',          'alignment',         'instruct_gpt.pdf'],
  // LoRA
  ['How does LoRA reduce the number of trainable parameters?', 'low-rank',      'lora.pdf'],
  ['What weights does LoRA keep frozen during training?',  'frozen',            'lora.pdf'],
  ['What is the rank hyperparameter in LoRA?',             'intrinsic',         'lora.pdf'],
  ['What is the inference overhead introduced by LoRA?',   'latency',           'lora.pdf'],
  // ViT
  ['How does ViT split images into tokens?',               'patch',             'vit.pdf'],
  ['What large dataset does ViT require for training?',    'JFT',               'vit.pdf'],
  ['How does ViT compare to CNNs on ImageNet?',            'convolutional',     'vit.pdf'],
  ['What position encoding does ViT use?',                 'positional',        'vit.pdf'],
  // GAN
  ['How does a GAN train the generator and discriminator?','adversarial',       'gan.pdf'],
  ['What are the two networks in a GAN?',                  'discriminator',     'gan.pdf'],
  ['What game-theoretic objective does GAN optimize?',     'minimax',           'gan.pdf'],
  ['What dataset does the original GAN evaluate on?',      'MNIST',             'gan.pdf'],
  // DDPM
  ['How does DDPM generate images through denoising?',     'denoising',         'ddpm.pdf'],
  ['What is the forward diffusion process in DDPM?',       'Markov',            'ddpm.pdf'],
  ['What neural architecture does DDPM use?',              'u-net',             'ddpm.pdf'],
  ['What image datasets does DDPM evaluate on?',           'CIFAR',             'ddpm.pdf'],
  // Word2Vec
  ['What is the skip-gram model in Word2Vec?',             'skip-gram',         'word2vec.pdf'],
  ['How does Word2Vec use negative sampling?',             'negative sampling', 'word2vec.pdf'],
  ['What is CBOW in Word2Vec?',                            'CBOW',              'word2vec.pdf'],
  ['What linguistic regularities does Word2Vec capture?',  'king',              'word2vec.pdf'],

  // ── hard semantic queries (vocabulary gap — BM25 expected to struggle) ──────
  ['What failure mode in deep networks motivated skip connections?',           'degradation',     'resnet.pdf'],
  ['How can a 175-billion-parameter model be adapted cheaply for a new task?', 'low-rank',        'lora.pdf'],
  ['How is a language model taught to be helpful using human preferences?',    'reinforcement',   'instruct_gpt.pdf'],
  ['What stops a generative network from producing low-quality outputs?',      'discriminator',   'gan.pdf'],
  ['How does removing noise step by step eventually produce a realistic image?','Markov',          'ddpm.pdf'],
  ['What approach lets a single model handle both translation and summarization?','text-to-text',  't5.pdf'],
  ['How do modern image transformers avoid processing individual pixels?',      'patch',           'vit.pdf'],
  ['What makes scientific pre-training better for technical retrieval tasks?',  'Semantic Scholar','scibert.pdf'],
  ['How does joint training on images and captions create a shared embedding space?','contrastive','clip.pdf'],
  ['Why does retrieval-augmented generation reduce factual errors?',            'retrieval',       'rag_lewis.pdf'],
];

// ── types ─────────────────────────────────────────────────────────────────────
interface IndexedChunk {
  text:      string;
  page:      number;
  pdfId:     string;
  embedding: number[];
}

// ── PDF extraction (pdfjs legacy — no worker needed in Node) ──────────────────
async function extractPDF(pdfPath: string): Promise<{ words: string[]; boundaries: [number, number][] }> {
  const { getDocument } = await import('pdfjs-dist/legacy/build/pdf.mjs') as typeof import('pdfjs-dist');
  const buf = await readFile(pdfPath);
  const doc = await getDocument({ data: new Uint8Array(buf), verbosity: 0 }).promise;

  const words: string[] = [];
  const boundaries: [number, number][] = [];

  for (let p = 1; p <= doc.numPages; p++) {
    const page    = await doc.getPage(p);
    const content = await page.getTextContent();
    const text    = (content.items as Array<{ str: string }>).map(i => i.str).join(' ');
    const ws      = text.trim().split(/\s+/).filter(Boolean);
    if (ws.length > 0) {
      boundaries.push([words.length, p]);
      words.push(...ws);
    }
  }

  return { words, boundaries };
}

// ── embedding helpers ─────────────────────────────────────────────────────────
type Embedder = Awaited<ReturnType<typeof pipeline>>;

async function embedBatch(embedder: Embedder, texts: string[]): Promise<number[][]> {
  // @ts-ignore — pipeline output shape varies; we know feature-extraction returns tensor
  const out = await embedder(texts, { pooling: 'mean', normalize: true });
  const [batchSize, embedDim] = (out as { dims: number[] }).dims;
  const data = (out as { data: Float32Array }).data;
  const result: number[][] = [];
  for (let i = 0; i < batchSize; i++) {
    result.push(Array.from(data.slice(i * embedDim, (i + 1) * embedDim)));
  }
  return result;
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function argsortDesc(arr: number[]): number[] {
  return Array.from({ length: arr.length }, (_, i) => i).sort((a, b) => arr[b] - arr[a]);
}

// ── main ──────────────────────────────────────────────────────────────────────
async function main() {
  // 1. load model
  process.stdout.write('Loading all-MiniLM-L6-v2 (q8)…');
  const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'q8' });
  console.log(' done');

  // 2. ingest all papers
  const papers = [
    'attention_is_all_you_need.pdf','bert.pdf','clip.pdf','ddpm.pdf',
    'dpr.pdf','gan.pdf','gpt3.pdf','instruct_gpt.pdf','lora.pdf',
    'rag_lewis.pdf','resnet.pdf','scibert.pdf','t5.pdf','vit.pdf','word2vec.pdf',
  ];

  const corpus: IndexedChunk[] = [];

  console.log('\nIngesting papers:');
  for (const paper of papers) {
    process.stdout.write(`  ${paper}…`);
    const { words, boundaries } = await extractPDF(join(PAPERS_DIR, paper));
    const chunks = chunkWords(words, boundaries);

    for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
      const batch = chunks.slice(i, i + BATCH_SIZE);
      const embs  = await embedBatch(embedder, batch.map(c => c.text));
      for (let j = 0; j < batch.length; j++) {
        corpus.push({ text: batch[j].text, page: batch[j].page, pdfId: paper, embedding: embs[j] });
      }
    }
    console.log(` ${chunks.length} chunks`);
  }
  console.log(`\nTotal corpus: ${corpus.length} chunks`);

  // 3. build Orama BM25 index (once, reused for all queries)
  type Doc = { text: string; idx: number };
  const db = await create({ schema: { text: 'string', idx: 'number' } as const });
  await insertMultiple(db, corpus.map((c, i): Doc => ({ text: c.text, idx: i })));

  // 4. eval
  const N_ORIG = 60; // first 60 = original eval.py questions; 61+ = hard semantic queries
  const hits: Record<number, { hybrid: number; bm25: number }> = {};
  const hitsOrig: Record<number, { hybrid: number; bm25: number }> = {};
  const hitsHard: Record<number, { hybrid: number; bm25: number }> = {};
  for (const k of EVAL_KS) {
    hits[k]     = { hybrid: 0, bm25: 0 };
    hitsOrig[k] = { hybrid: 0, bm25: 0 };
    hitsHard[k] = { hybrid: 0, bm25: 0 };
  }

  const n = CASES.length;
  console.log('\n' + '-'.repeat(80));
  console.log(`${'#'.padEnd(4)} ${'Hybrid@10'.padEnd(12)} ${'BM25@10'.padEnd(10)}  Question`);
  console.log('-'.repeat(80));

  for (let i = 0; i < CASES.length; i++) {
    const [question, keyword, pdfId] = CASES[i];
    const kw    = keyword.toLowerCase();
    const isOrig = i < N_ORIG;

    const [queryEmb] = await embedBatch(embedder, [question]);

    for (const k of EVAL_KS) {
      const pool = Math.max(50, 3 * k);

      // dense: cosine over full corpus (embeddings are normalised → dot = cosine)
      const sims       = corpus.map(c => dot(c.embedding, queryEmb));
      const denseOrder = argsortDesc(sims).slice(0, pool);
      const denseRank  = new Map(denseOrder.map((idx, rank) => [idx, rank]));

      // BM25 (threshold:1 = all query tokens must match — mirrors browser main.ts)
      const bm25Res  = await search(db, { term: question, properties: ['text'], limit: pool, threshold: 1 });
      const bm25Rank = new Map(bm25Res.hits.map((h, rank) => [(h.document as Doc).idx, rank]));

      // RRF 2:1 dense:BM25
      const fused = new Map<number, number>();
      for (const [idx, rank] of denseRank) fused.set(idx, (fused.get(idx) ?? 0) + 2.0 / (RRF_K + rank + 1));
      for (const [idx, rank] of bm25Rank) fused.set(idx, (fused.get(idx) ?? 0) + 1.0 / (RRF_K + rank + 1));

      const topHybrid = [...fused.entries()].sort((a, b) => b[1] - a[1]).slice(0, k).map(([idx]) => corpus[idx]);
      const topBM25   = bm25Res.hits.slice(0, k).map(h => corpus[(h.document as Doc).idx]);

      const hybridHit = topHybrid.some(c => c.text.toLowerCase().includes(kw) && c.pdfId === pdfId);
      const bm25Hit   = topBM25.some(c => c.text.toLowerCase().includes(kw)   && c.pdfId === pdfId);

      hits[k].hybrid     += hybridHit ? 1 : 0;
      hits[k].bm25       += bm25Hit   ? 1 : 0;
      if (isOrig) { hitsOrig[k].hybrid += hybridHit ? 1 : 0; hitsOrig[k].bm25 += bm25Hit ? 1 : 0; }
      else        { hitsHard[k].hybrid += hybridHit ? 1 : 0; hitsHard[k].bm25 += bm25Hit ? 1 : 0; }

      if (k === 10) {
        const h = hybridHit ? 'PASS' : 'FAIL';
        const b = bm25Hit   ? 'PASS' : 'FAIL';
        const tag = isOrig ? '' : ' [hard]';
        console.log(`${String(i + 1).padEnd(4)} ${h.padEnd(12)} ${b.padEnd(10)}  ${question}${tag}`);
      }
    }
  }

  // 5. summary
  const printTable = (label: string, hitsMap: typeof hits, total: number) => {
    console.log(`\n── ${label} (n=${total})`);
    console.log(`${'Metric'.padEnd(22)} ${'Hybrid RRF (2:1)'.padStart(18)} ${'BM25 only'.padStart(14)}`);
    console.log('-'.repeat(56));
    for (const k of EVAL_KS) {
      const h = hitsMap[k].hybrid, b = hitsMap[k].bm25;
      console.log(
        `${'Recall@' + k}${' '.repeat(22 - ('Recall@' + k).length)}` +
        `${(h / total * 100).toFixed(1).padStart(16)}%  (${h}/${total})` +
        `${(b / total * 100).toFixed(1).padStart(12)}%  (${b}/${total})`
      );
    }
  };

  console.log('\n' + '='.repeat(80));
  printTable('Original 60 questions (same as eval.py)', hitsOrig, N_ORIG);
  printTable('Hard 10 questions (semantic vocabulary gap)', hitsHard, n - N_ORIG);
  printTable('All 70 questions', hits, n);
  console.log('\n' + '='.repeat(80));
  console.log('Note: BM25 threshold:1 (all query tokens must match) — mirrors browser main.ts.');
}

main().catch(err => { console.error(err); process.exit(1); });
