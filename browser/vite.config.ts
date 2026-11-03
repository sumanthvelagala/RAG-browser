import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [
    {
      // The ort-wasm-simd-threaded.asyncify.wasm is 26 MB, exceeding Cloudflare
      // Pages' 25 MB per-file limit. We remove it from the bundle and load it
      // from CDN at runtime via env.backends.onnx.wasm.wasmPaths.
      name: 'remove-ort-asyncify-wasm',
      generateBundle(_options, bundle) {
        for (const key of Object.keys(bundle)) {
          if (key.includes('ort-wasm-simd-threaded.asyncify')) {
            delete bundle[key];
          }
        }
      },
    },
  ],
});
