/* Offline cache for the PDF search app.
 *
 *  - ONNX runtime WASM from jsDelivr (version-pinned to 1.31.1): cache-first
 *  - Same-origin app shell (HTML/JS/CSS/icons):                   stale-while-revalidate
 *  - Model weights (Hugging Face): handled by Transformers.js' own Cache API
 *    store, deliberately not touched here.
 *
 * Bump CACHE_VERSION to drop old entries on the next deploy.
 * When upgrading onnxruntime-web, also update ORT_VERSION below and in
 * embedder.worker.ts so they stay in sync.
 */
const CACHE_VERSION = 'v1';
const RUNTIME_CACHE = `ort-runtime-${CACHE_VERSION}`;
const SHELL_CACHE   = `app-shell-${CACHE_VERSION}`;
const ORT_VERSION   = '1.31.1';

self.addEventListener('install', () => self.skipWaiting());

self.addEventListener('activate', event => {
  event.waitUntil(
    (async () => {
      const keep = new Set([RUNTIME_CACHE, SHELL_CACHE]);
      for (const key of await caches.keys()) {
        if (!keep.has(key) && (key.startsWith('ort-runtime-') || key.startsWith('app-shell-'))) {
          await caches.delete(key);
        }
      }
      await self.clients.claim();
    })(),
  );
});

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const hit   = await cache.match(request);
  if (hit) return hit;
  const res = await fetch(request);
  if (res.ok) cache.put(request, res.clone());
  return res;
}

async function staleWhileRevalidate(request, cacheName) {
  const cache   = await caches.open(cacheName);
  const hit     = await cache.match(request);
  const network = fetch(request)
    .then(res => { if (res.ok) cache.put(request, res.clone()); return res; })
    .catch(() => null);
  return hit || (await network) || Response.error();
}

self.addEventListener('fetch', event => {
  const { request } = event;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);

  // Cache-first: pinned ORT runtime from CDN.
  // Only match the exact pinned version so a future upgrade doesn't serve stale WASM.
  if (
    url.hostname === 'cdn.jsdelivr.net' &&
    url.pathname.includes(`/npm/onnxruntime-web@${ORT_VERSION}/dist/`)
  ) {
    event.respondWith(cacheFirst(request, RUNTIME_CACHE));
    return;
  }

  // Stale-while-revalidate: app shell assets.
  // On navigate, fall back to cached index.html when fully offline.
  if (url.origin === self.location.origin) {
    event.respondWith(
      staleWhileRevalidate(request, SHELL_CACHE).then(res =>
        res.type === 'error' && request.mode === 'navigate'
          ? caches.match(new URL('./', self.location).href).then(r => r || res)
          : res,
      ),
    );
  }
});
