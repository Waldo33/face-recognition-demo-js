import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.mjs";
import { l2Normalize } from "./math.js";

const CROP_SIZE = 112;

const cropCanvas = createCanvas();
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });
const tensorCanvas = createCanvas();
const tensorCtx = tensorCanvas.getContext("2d", { willReadFrequently: true });

export function createRecognizerSession(modelUrl) {
  return ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all"
  });
}

export async function extractEmbedding(session, source, faceBox, options = {}) {
  const padding = options.padding ?? 0.25;
  const croppedFace = cropFace(source, faceBox, padding);
  const tensor = faceToTensor(session, croppedFace);
  const outputMap = await session.run({ [session.inputNames[0]]: tensor });
  const outputTensor = pickEmbeddingTensor(outputMap);
  if (!outputTensor) {
    throw new Error("Recognizer output tensor is empty.");
  }

  const embedding = l2Normalize(outputTensor.data);
  validateEmbedding(embedding);
  return embedding;
}

function cropFace(source, faceBox, padding) {
  const { width: srcW, height: srcH } = getSourceSize(source);
  const centerX = faceBox.x + faceBox.width / 2;
  const centerY = faceBox.y + faceBox.height / 2;
  const side = Math.max(faceBox.width, faceBox.height) * (1 + padding * 2);

  const x = clamp(centerX - side / 2, 0, srcW - 1);
  const y = clamp(centerY - side / 2, 0, srcH - 1);
  const w = clamp(side, 1, srcW - x);
  const h = clamp(side, 1, srcH - y);

  cropCanvas.width = CROP_SIZE;
  cropCanvas.height = CROP_SIZE;
  cropCtx.clearRect(0, 0, CROP_SIZE, CROP_SIZE);
  cropCtx.drawImage(source, x, y, w, h, 0, 0, CROP_SIZE, CROP_SIZE);
  return cropCanvas;
}

function faceToTensor(session, source) {
  const dims = session.inputMetadata?.[session.inputNames[0]]?.dimensions;
  const isNhwc = Array.isArray(dims) && dims[3] === 3;

  // Use a separate canvas for tensor prep. If source===cropCanvas, clearing the same
  // canvas before draw would erase the cropped face and produce near-constant embeddings.
  tensorCanvas.width = CROP_SIZE;
  tensorCanvas.height = CROP_SIZE;
  tensorCtx.clearRect(0, 0, CROP_SIZE, CROP_SIZE);
  tensorCtx.drawImage(source, 0, 0, CROP_SIZE, CROP_SIZE);
  const { data } = tensorCtx.getImageData(0, 0, CROP_SIZE, CROP_SIZE);
  const area = CROP_SIZE * CROP_SIZE;

  if (isNhwc) {
    const nhwc = new Float32Array(area * 3);
    for (let i = 0; i < area; i += 1) {
      nhwc[i * 3] = (data[i * 4] - 127.5) / 127.5;
      nhwc[i * 3 + 1] = (data[i * 4 + 1] - 127.5) / 127.5;
      nhwc[i * 3 + 2] = (data[i * 4 + 2] - 127.5) / 127.5;
    }
    return new ort.Tensor("float32", nhwc, [1, CROP_SIZE, CROP_SIZE, 3]);
  }

  const nchw = new Float32Array(area * 3);
  for (let i = 0; i < area; i += 1) {
    const r = (data[i * 4] - 127.5) / 127.5;
    const g = (data[i * 4 + 1] - 127.5) / 127.5;
    const b = (data[i * 4 + 2] - 127.5) / 127.5;
    nchw[i] = r;
    nchw[area + i] = g;
    nchw[area * 2 + i] = b;
  }
  return new ort.Tensor("float32", nchw, [1, 3, CROP_SIZE, CROP_SIZE]);
}

function getSourceSize(source) {
  if ("videoWidth" in source && source.videoWidth > 0) {
    return { width: source.videoWidth, height: source.videoHeight };
  }
  if ("naturalWidth" in source && source.naturalWidth > 0) {
    return { width: source.naturalWidth, height: source.naturalHeight };
  }
  return { width: source.width, height: source.height };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function createCanvas() {
  if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(1, 1);
  }
  if (typeof document !== "undefined") {
    return document.createElement("canvas");
  }
  throw new Error("No canvas implementation available.");
}

function pickEmbeddingTensor(outputMap) {
  const tensors = Object.values(outputMap);
  if (!tensors.length) {
    return null;
  }

  // Take the most likely embedding output: largest vector-like tensor.
  let best = null;
  for (const tensor of tensors) {
    const size = tensor?.data?.length || 0;
    if (!size) {
      continue;
    }
    if (!best || size > best.data.length) {
      best = tensor;
    }
  }

  if (!best || best.data.length < 64) {
    return null;
  }
  return best;
}

function validateEmbedding(embedding) {
  if (!embedding || embedding.length < 64) {
    throw new Error("Recognizer returned invalid embedding length.");
  }

  // Detect degenerate outputs where all values are almost equal.
  const mean = embedding.reduce((sum, value) => sum + value, 0) / embedding.length;
  let variance = 0;
  for (let i = 0; i < embedding.length; i += 1) {
    const diff = embedding[i] - mean;
    variance += diff * diff;
  }
  variance /= embedding.length;

  if (variance < 1e-6) {
    throw new Error("Recognizer produced degenerate embedding.");
  }
}
