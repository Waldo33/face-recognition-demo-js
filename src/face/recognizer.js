import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.mjs";
import { l2Normalize } from "./math.js";

const CROP_SIZE = 112;

const cropCanvas = document.createElement("canvas");
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

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
  const outputTensor = Object.values(outputMap)[0];
  if (!outputTensor) {
    throw new Error("Recognizer output tensor is empty.");
  }
  return l2Normalize(outputTensor.data);
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

  cropCtx.clearRect(0, 0, CROP_SIZE, CROP_SIZE);
  cropCtx.drawImage(source, 0, 0, CROP_SIZE, CROP_SIZE);
  const { data } = cropCtx.getImageData(0, 0, CROP_SIZE, CROP_SIZE);
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
