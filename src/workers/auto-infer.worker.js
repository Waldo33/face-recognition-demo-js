import { bestCosineMatch, getRecommendedThreshold } from "../face/math.js";
import { extractEmbedding, createRecognizerSession } from "../face/recognizer.js";
import { createDetectorSession, detectFaces, getPrimaryFace } from "../face/ultraface.js";

const state = {
  detectorSession: null,
  recognizerSession: null,
  people: [],
  sourceCanvas: null,
  sourceCtx: null,
  lastPrimaryFace: null,
  lastDetectAt: 0
};

state.sourceCanvas = createCanvas(1, 1);
state.sourceCtx = state.sourceCanvas.getContext("2d", { willReadFrequently: true });

self.onmessage = async (event) => {
  const { requestId, type, payload } = event.data || {};
  try {
    let result = null;
    if (type === "init") {
      result = await handleInit(payload);
    } else if (type === "setPeople") {
      result = handleSetPeople(payload);
    } else if (type === "resetTracking") {
      result = handleResetTracking();
    } else if (type === "processFrame") {
      result = await handleProcessFrame(payload);
    } else {
      throw new Error(`Неподдерживаемый тип сообщения воркера: ${type}`);
    }

    self.postMessage({ requestId, ok: true, payload: result });
  } catch (error) {
    self.postMessage({
      requestId,
      ok: false,
      error: error?.message || "Ошибка обработки в воркере."
    });
  }
};

async function handleInit(payload) {
  if (state.detectorSession && state.recognizerSession) {
    return { ready: true };
  }

  const modelPaths = payload?.modelPaths || {};
  state.detectorSession = await createDetectorSession(modelPaths.detector);
  state.recognizerSession = await createRecognizerSession(modelPaths.recognizer);
  return { ready: true };
}

function handleSetPeople(payload) {
  const people = Array.isArray(payload?.people) ? payload.people : [];
  state.people = people
    .filter((person) => person && typeof person.personId === "string")
    .map((person) => ({
      personId: person.personId,
      firstName: person.firstName || "",
      lastName: person.lastName || "",
      sampleCount: person.sampleCount || 0,
      embeddings: Array.isArray(person.embeddings)
        ? person.embeddings
            .map((embedding) => toFloat32(embedding))
            .filter((embedding) => embedding.length > 0)
        : []
    }))
    .filter((person) => person.embeddings.length > 0);

  return { peopleCount: state.people.length };
}

function handleResetTracking() {
  state.lastPrimaryFace = null;
  state.lastDetectAt = 0;
  return { ok: true };
}

async function handleProcessFrame(payload) {
  if (!state.detectorSession || !state.recognizerSession) {
    throw new Error("Сессии воркера не инициализированы.");
  }

  const startedAt = performance.now();
  const imageBitmap = payload?.imageBitmap;
  if (!imageBitmap) {
    throw new Error("В payload processFrame отсутствует imageBitmap.");
  }

  const maxSide = payload?.maxSide ?? 480;
  drawBitmapToSource(imageBitmap, maxSide);
  if (typeof imageBitmap.close === "function") {
    imageBitmap.close();
  }

  const now = payload?.timestamp ?? performance.now();
  const detectIntervalMs = payload?.detectIntervalMs ?? 900;
  const runDetection =
    !state.lastPrimaryFace || now - state.lastDetectAt >= detectIntervalMs;

  let faces = [];
  let primaryFace = null;
  if (runDetection) {
    faces = await detectFaces(state.detectorSession, state.sourceCanvas, {
      scoreThreshold: 0.7,
      iouThreshold: 0.3,
      maxResults: 5
    });
    primaryFace = getPrimaryFace(faces);
    state.lastPrimaryFace = primaryFace ? cloneFace(primaryFace) : null;
    state.lastDetectAt = now;
  } else if (state.lastPrimaryFace) {
    primaryFace = clampFaceToSource(state.lastPrimaryFace);
    faces = primaryFace ? [primaryFace] : [];
  }

  const latencyMs = performance.now() - startedAt;

  if (!primaryFace) {
    state.lastPrimaryFace = null;
    return {
      latencyMs,
      ranDetection: runDetection,
      frameWidth: state.sourceCanvas.width,
      frameHeight: state.sourceCanvas.height,
      faces: [],
      primaryFace: null,
      noFace: true,
      best: null,
      threshold: payload?.baseThreshold ?? 0.55,
      minGap: payload?.minGap ?? 0
    };
  }

  const embedding = await extractEmbedding(
    state.recognizerSession,
    state.sourceCanvas,
    primaryFace,
    { padding: 0.3 }
  );

  const baseThreshold = payload?.baseThreshold ?? 0.55;
  const threshold = getRecommendedThreshold(state.people.length, baseThreshold);
  const minGap = state.people.length > 1 ? payload?.minGap ?? 0.03 : 0;
  const rawBest = bestCosineMatch(embedding, state.people, threshold, { minGap });
  const best = toCompactBest(rawBest);

  return {
    latencyMs: performance.now() - startedAt,
    ranDetection: runDetection,
    frameWidth: state.sourceCanvas.width,
    frameHeight: state.sourceCanvas.height,
    faces: primaryFace ? [primaryFace] : [],
    primaryFace,
    noFace: false,
    best,
    threshold,
    minGap
  };
}

function drawBitmapToSource(imageBitmap, maxSide) {
  const srcW = imageBitmap.width;
  const srcH = imageBitmap.height;
  const scale = Math.min(1, maxSide / Math.max(srcW, srcH));
  const width = Math.max(1, Math.round(srcW * scale));
  const height = Math.max(1, Math.round(srcH * scale));

  if (state.sourceCanvas.width !== width || state.sourceCanvas.height !== height) {
    state.sourceCanvas.width = width;
    state.sourceCanvas.height = height;
  }

  state.sourceCtx.clearRect(0, 0, width, height);
  state.sourceCtx.drawImage(imageBitmap, 0, 0, width, height);
}

function toCompactBest(best) {
  if (!best?.person) {
    return best;
  }

  return {
    matched: best.matched,
    score: best.score,
    sampleIndex: best.sampleIndex,
    gap: best.gap,
    person: {
      personId: best.person.personId,
      firstName: best.person.firstName || "",
      lastName: best.person.lastName || "",
      sampleCount: best.person.sampleCount || best.person.embeddings?.length || 0
    }
  };
}

function toFloat32(value) {
  if (value instanceof Float32Array) {
    return value;
  }
  if (value instanceof ArrayBuffer) {
    return new Float32Array(value);
  }
  if (ArrayBuffer.isView(value)) {
    return new Float32Array(value.buffer, value.byteOffset, value.byteLength / 4);
  }
  if (Array.isArray(value)) {
    return Float32Array.from(value);
  }
  return new Float32Array();
}

function createCanvas(width, height) {
  if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(width, height);
  }
  throw new Error("OffscreenCanvas недоступен в воркере.");
}

function cloneFace(face) {
  return {
    x: face.x,
    y: face.y,
    width: face.width,
    height: face.height,
    score: face.score
  };
}

function clampFaceToSource(face) {
  const width = state.sourceCanvas.width;
  const height = state.sourceCanvas.height;
  if (!width || !height) {
    return null;
  }

  const x = clamp(face.x, 0, width - 1);
  const y = clamp(face.y, 0, height - 1);
  const w = clamp(face.width, 1, width - x);
  const h = clamp(face.height, 1, height - y);
  if (w < 4 || h < 4) {
    return null;
  }

  return { x, y, width: w, height: h, score: face.score };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
