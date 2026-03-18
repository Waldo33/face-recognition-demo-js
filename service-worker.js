const CACHE_VERSION = "v1.0.0";
const CACHE_PREFIX = "face-id-demo";
const STATIC_CACHE = `${CACHE_PREFIX}-static-${CACHE_VERSION}`;
const MODEL_CACHE = `${CACHE_PREFIX}-models-${CACHE_VERSION}`;
const RUNTIME_CACHE = `${CACHE_PREFIX}-runtime-${CACHE_VERSION}`;
const CDN_CACHE = `${CACHE_PREFIX}-cdn-${CACHE_VERSION}`;

const APP_SHELL_PATHS = ["./", "./index.html", "./src/style.css", "./src/main.js"];
const MODEL_PATHS = [
  "./models/version-RFB-320.onnx",
  "./models/w600k_mbf.onnx",
  "./public/models/version-RFB-320.onnx",
  "./public/models/w600k_mbf.onnx"
];
const CDN_PREFIXES = [
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@"
];
const CDN_ASSETS = [
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.mjs",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd.jsep.wasm",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-threaded.jsep.wasm",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.jsep.wasm",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd.wasm",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-threaded.wasm",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.wasm"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const staticCache = await caches.open(STATIC_CACHE);
      const modelCache = await caches.open(MODEL_CACHE);
      const cdnCache = await caches.open(CDN_CACHE);

      await cacheBestEffort(staticCache, APP_SHELL_PATHS);
      await cacheBestEffort(modelCache, MODEL_PATHS);
      await cacheBestEffort(cdnCache, CDN_ASSETS);

      await self.skipWaiting();
    })()
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys.map((key) => {
          if (!key.startsWith(CACHE_PREFIX)) {
            return Promise.resolve();
          }
          if (key === STATIC_CACHE || key === MODEL_CACHE || key === RUNTIME_CACHE || key === CDN_CACHE) {
            return Promise.resolve();
          }
          return caches.delete(key);
        })
      );
      await self.clients.claim();
    })()
  );
});

self.addEventListener("message", (event) => {
  if (event.data?.type === "SKIP_WAITING") {
    self.skipWaiting();
  }
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") {
    return;
  }

  const url = new URL(request.url);
  if (!url.protocol.startsWith("http")) {
    return;
  }

  if (request.mode === "navigate") {
    event.respondWith(networkFirstWithOfflineFallback(request, STATIC_CACHE));
    return;
  }

  const isSameOrigin = url.origin === self.location.origin;
  const isModelRequest = isSameOrigin && url.pathname.includes("/models/") && url.pathname.endsWith(".onnx");
  if (isModelRequest) {
    event.respondWith(cacheFirst(request, MODEL_CACHE));
    return;
  }

  const isOnnxCdn = CDN_PREFIXES.some((prefix) => request.url.startsWith(prefix));
  if (isOnnxCdn) {
    event.respondWith(staleWhileRevalidate(request, CDN_CACHE));
    return;
  }

  if (isSameOrigin) {
    event.respondWith(staleWhileRevalidate(request, RUNTIME_CACHE));
  }
});

async function cacheBestEffort(cache, paths) {
  await Promise.all(
    paths.map(async (path) => {
      const absoluteUrl = new URL(path, self.registration.scope).toString();
      try {
        const response = await fetch(absoluteUrl, { cache: "no-cache" });
        if (response.ok || response.type === "opaque") {
          await cache.put(absoluteUrl, response.clone());
        }
      } catch {
        // Ignore warmup errors; runtime caching will fill missing entries.
      }
    })
  );
}

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) {
    return cached;
  }

  const response = await fetch(request);
  if (response.ok || response.type === "opaque") {
    await cache.put(request, response.clone());
  }
  return response;
}

async function staleWhileRevalidate(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);

  const networkPromise = fetch(request)
    .then(async (response) => {
      if (response.ok || response.type === "opaque") {
        await cache.put(request, response.clone());
      }
      return response;
    })
    .catch(() => null);

  if (cached) {
    void networkPromise;
    return cached;
  }

  const networkResponse = await networkPromise;
  if (networkResponse) {
    return networkResponse;
  }

  return new Response("Offline", { status: 503, statusText: "Offline" });
}

async function networkFirstWithOfflineFallback(request, cacheName) {
  const cache = await caches.open(cacheName);
  try {
    const response = await fetch(request);
    if (response.ok || response.type === "opaque") {
      await cache.put(request, response.clone());
      const appShellIndex = new URL("./index.html", self.registration.scope).toString();
      await cache.put(appShellIndex, response.clone());
    }
    return response;
  } catch {
    const cachedNavigation = await cache.match(request);
    if (cachedNavigation) {
      return cachedNavigation;
    }

    const appShellIndex = new URL("./index.html", self.registration.scope).toString();
    const cachedIndex = await cache.match(appShellIndex);
    if (cachedIndex) {
      return cachedIndex;
    }

    return new Response("Offline", { status: 503, statusText: "Offline" });
  }
}
