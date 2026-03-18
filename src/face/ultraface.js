import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.mjs";

const ULTRAFACE_CONFIG = {
  inputWidth: 320,
  inputHeight: 240,
  centerVariance: 0.1,
  sizeVariance: 0.2,
  minBoxes: [
    [10, 16, 24],
    [32, 48],
    [64, 96],
    [128, 192, 256]
  ],
  strides: [8, 16, 32, 64]
};

const sharedCanvas = createCanvas();
const sharedCtx = sharedCanvas.getContext("2d", { willReadFrequently: true });
const priors = generatePriors(ULTRAFACE_CONFIG);

export async function detectFaces(session, source, options = {}) {
  const scoreThreshold = options.scoreThreshold ?? 0.7;
  const iouThreshold = options.iouThreshold ?? 0.3;
  const maxResults = options.maxResults ?? 5;

  const tensor = imageToTensor(
    source,
    ULTRAFACE_CONFIG.inputWidth,
    ULTRAFACE_CONFIG.inputHeight
  );

  const feeds = { [session.inputNames[0]]: tensor };
  const outputMap = await session.run(feeds);
  const tensors = Object.values(outputMap);
  const boxesTensor = tensors.find((item) => item.dims[item.dims.length - 1] === 4);
  const scoresTensor = tensors.find((item) => item.dims[item.dims.length - 1] === 2);

  if (!boxesTensor || !scoresTensor) {
    throw new Error("Не удалось распознать выходы модели UltraFace.");
  }

  const sourceSize = getSourceSize(source);
  const candidates = decodeDetections(
    boxesTensor.data,
    scoresTensor.data,
    sourceSize.width,
    sourceSize.height,
    scoreThreshold
  );
  const picked = nonMaxSuppression(candidates, iouThreshold, maxResults);
  return picked;
}

export function getPrimaryFace(faces) {
  if (!faces.length) {
    return null;
  }
  return [...faces].sort((a, b) => b.width * b.height - a.width * a.height)[0];
}

export function createDetectorSession(modelUrl) {
  return ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all"
  });
}

function imageToTensor(source, width, height) {
  sharedCanvas.width = width;
  sharedCanvas.height = height;
  sharedCtx.clearRect(0, 0, width, height);
  sharedCtx.drawImage(source, 0, 0, width, height);
  const { data } = sharedCtx.getImageData(0, 0, width, height);

  const area = width * height;
  const chw = new Float32Array(area * 3);
  for (let i = 0; i < area; i += 1) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    chw[i] = (r - 127) / 128;
    chw[area + i] = (g - 127) / 128;
    chw[area * 2 + i] = (b - 127) / 128;
  }

  return new ort.Tensor("float32", chw, [1, 3, height, width]);
}

function decodeDetections(
  rawBoxes,
  rawScores,
  sourceWidth,
  sourceHeight,
  scoreThreshold
) {
  const detections = [];

  for (let i = 0; i < priors.length; i += 1) {
    const score = rawScores[i * 2 + 1];
    if (score < scoreThreshold) {
      continue;
    }

    const prior = priors[i];
    const offset = i * 4;

    const centerX =
      rawBoxes[offset] * ULTRAFACE_CONFIG.centerVariance * prior[2] + prior[0];
    const centerY =
      rawBoxes[offset + 1] * ULTRAFACE_CONFIG.centerVariance * prior[3] + prior[1];
    const boxW =
      Math.exp(rawBoxes[offset + 2] * ULTRAFACE_CONFIG.sizeVariance) * prior[2];
    const boxH =
      Math.exp(rawBoxes[offset + 3] * ULTRAFACE_CONFIG.sizeVariance) * prior[3];

    const xMin = clamp((centerX - boxW / 2) * sourceWidth, 0, sourceWidth);
    const yMin = clamp((centerY - boxH / 2) * sourceHeight, 0, sourceHeight);
    const xMax = clamp((centerX + boxW / 2) * sourceWidth, 0, sourceWidth);
    const yMax = clamp((centerY + boxH / 2) * sourceHeight, 0, sourceHeight);

    const width = xMax - xMin;
    const height = yMax - yMin;
    if (width < 4 || height < 4) {
      continue;
    }

    detections.push({
      x: xMin,
      y: yMin,
      width,
      height,
      score
    });
  }

  return detections.sort((a, b) => b.score - a.score);
}

function generatePriors(config) {
  const priorsList = [];
  for (let level = 0; level < config.strides.length; level += 1) {
    const stride = config.strides[level];
    const featureMapWidth = Math.ceil(config.inputWidth / stride);
    const featureMapHeight = Math.ceil(config.inputHeight / stride);

    for (let y = 0; y < featureMapHeight; y += 1) {
      for (let x = 0; x < featureMapWidth; x += 1) {
        for (const box of config.minBoxes[level]) {
          const cx = (x + 0.5) * stride / config.inputWidth;
          const cy = (y + 0.5) * stride / config.inputHeight;
          const sx = box / config.inputWidth;
          const sy = box / config.inputHeight;
          priorsList.push([cx, cy, sx, sy]);
        }
      }
    }
  }
  return priorsList;
}

function nonMaxSuppression(boxes, iouThreshold, maxResults) {
  const results = [];
  const queue = [...boxes];

  while (queue.length && results.length < maxResults) {
    const candidate = queue.shift();
    results.push(candidate);

    for (let i = queue.length - 1; i >= 0; i -= 1) {
      if (iou(candidate, queue[i]) > iouThreshold) {
        queue.splice(i, 1);
      }
    }
  }

  return results;
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const intersection = interW * interH;
  const union = a.width * a.height + b.width * b.height - intersection;
  return union <= 0 ? 0 : intersection / union;
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
  throw new Error("В окружении недоступна реализация canvas.");
}
