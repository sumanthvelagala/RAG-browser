/**
 * Embedding Web Worker — runs all-MiniLM-L6-v2 (quantized int8, ~22 MB)
 * off the main thread so the UI never freezes during indexing or querying.
 */
import { pipeline, env, FeatureExtractionPipeline } from '@huggingface/transformers';

// The asyncify WASM (26 MB) exceeds Cloudflare Pages' 25 MB per-file limit.
// Load it from CDN so the local copy is excluded from the dist bundle.
if (env.backends.onnx.wasm) {
  env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
}

let embedder: FeatureExtractionPipeline | null = null;

async function getEmbedder(): Promise<FeatureExtractionPipeline> {
  if (!embedder) {
    self.postMessage({ type: 'model_loading' });
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      dtype: 'q8',
      progress_callback: (p: { status: string; progress?: number }) => {
        if (p.status === 'progress') {
          self.postMessage({ type: 'model_progress', progress: p.progress ?? 0 });
        }
      },
    });
    self.postMessage({ type: 'model_ready' });
  }
  return embedder;
}

async function embed(texts: string[]): Promise<number[][]> {
  const model = await getEmbedder();
  const out = await model(texts, { pooling: 'mean', normalize: true });
  // out.data is a flat Float32Array [text0_dim0, text0_dim1, ..., text1_dim0, ...]
  const dims = out.dims; // [batchSize, 384]
  const batchSize = dims[0];
  const embedDim = dims[1];
  const result: number[][] = [];
  for (let i = 0; i < batchSize; i++) {
    result.push(Array.from(out.data.slice(i * embedDim, (i + 1) * embedDim) as Float32Array));
  }
  return result;
}

self.onmessage = async (e: MessageEvent) => {
  const { id, texts } = e.data;
  try {
    const embeddings = await embed(texts);
    self.postMessage({ type: 'embed_result', id, embeddings });
  } catch (err) {
    self.postMessage({ type: 'embed_error', id, error: String(err) });
  }
};
