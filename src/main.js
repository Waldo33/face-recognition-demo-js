import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.mjs";
import { bestCosineMatch } from "./face/math.js";
import { extractEmbedding, createRecognizerSession } from "./face/recognizer.js";
import {
  clearPeopleCache,
  listPeople,
  removePerson,
  upsertPerson
} from "./face/storage.js";
import { createDetectorSession, detectFaces, getPrimaryFace } from "./face/ultraface.js";

const MODEL_PATHS = {
  detector: "/models/version-RFB-320.onnx",
  recognizer: "/models/w600k_mbf.onnx"
};

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
ort.env.wasm.simd = true;

const elements = {
  video: document.getElementById("camera"),
  frameCanvas: document.getElementById("frameCanvas"),
  imageInput: document.getElementById("imageInput"),
  startCameraBtn: document.getElementById("startCameraBtn"),
  stopCameraBtn: document.getElementById("stopCameraBtn"),
  captureBtn: document.getElementById("captureBtn"),
  status: document.getElementById("status"),
  personIdInput: document.getElementById("personIdInput"),
  firstNameInput: document.getElementById("firstNameInput"),
  lastNameInput: document.getElementById("lastNameInput"),
  enrollBtn: document.getElementById("enrollBtn"),
  identifyBtn: document.getElementById("identifyBtn"),
  thresholdInput: document.getElementById("thresholdInput"),
  result: document.getElementById("result"),
  peopleList: document.getElementById("peopleList"),
  clearCacheBtn: document.getElementById("clearCacheBtn")
};

const appState = {
  stream: null,
  detectorSession: null,
  recognizerSession: null,
  busy: false,
  sourceCanvas: document.createElement("canvas"),
  sourceCtx: null
};
appState.sourceCtx = appState.sourceCanvas.getContext("2d");

function init() {
  bindEvents();
  renderPeopleList();
  setStatus("Load a frame from camera or image.");
  setResult("No identification run yet.");
}

function bindEvents() {
  elements.startCameraBtn.addEventListener("click", () => runSafe(startCamera));
  elements.stopCameraBtn.addEventListener("click", stopCamera);
  elements.captureBtn.addEventListener("click", () => runSafe(captureFrame));
  elements.enrollBtn.addEventListener("click", () => runSafe(enrollFace));
  elements.identifyBtn.addEventListener("click", () => runSafe(identifyFace));
  elements.clearCacheBtn.addEventListener("click", clearCache);
  elements.imageInput.addEventListener("change", (event) => runSafe(() => loadImage(event)));
}

async function runSafe(task) {
  if (appState.busy) {
    return;
  }
  appState.busy = true;
  toggleActions(true);
  try {
    await task();
  } catch (error) {
    setStatus(error.message || "Operation failed.");
  } finally {
    appState.busy = false;
    toggleActions(false);
  }
}

function toggleActions(disabled) {
  elements.captureBtn.disabled = disabled;
  elements.enrollBtn.disabled = disabled;
  elements.identifyBtn.disabled = disabled;
}

async function startCamera() {
  if (appState.stream) {
    setStatus("Camera is already active.");
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Camera API is not available in this browser.");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
    audio: false
  });
  appState.stream = stream;
  elements.video.srcObject = stream;
  await elements.video.play();
  setStatus("Camera started. Press Capture Frame.");
}

function stopCamera() {
  if (!appState.stream) {
    setStatus("Camera already stopped.");
    return;
  }

  for (const track of appState.stream.getTracks()) {
    track.stop();
  }
  appState.stream = null;
  elements.video.srcObject = null;
  setStatus("Camera stopped.");
}

function captureFrame() {
  if (!appState.stream || !elements.video.videoWidth) {
    throw new Error("Start the camera first.");
  }
  setSourceFrom(elements.video);
  renderFrame();
  setStatus("Frame captured.");
}

async function loadImage(event) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }

  const imageUrl = URL.createObjectURL(file);
  try {
    const image = await loadImageElement(imageUrl);
    setSourceFrom(image);
    renderFrame();
    setStatus(`Image loaded: ${file.name}`);
  } finally {
    URL.revokeObjectURL(imageUrl);
    elements.imageInput.value = "";
  }
}

async function enrollFace() {
  const personId = elements.personIdInput.value.trim();
  const firstName = elements.firstNameInput.value.trim();
  const lastName = elements.lastNameInput.value.trim();

  if (!personId) {
    throw new Error("Person ID is required.");
  }
  if (!hasSourceFrame()) {
    throw new Error("Capture a frame or upload an image first.");
  }

  await ensureModels();
  setStatus("Detecting face and creating embedding...");

  const { embedding } = await detectAndEmbed();
  upsertPerson({ personId, firstName, lastName, embedding });
  renderPeopleList();
  setStatus(`Saved embedding for "${personId}" in local cache.`);
}

async function identifyFace() {
  if (!hasSourceFrame()) {
    throw new Error("Capture a frame or upload an image first.");
  }
  const people = listPeople();
  if (!people.length) {
    throw new Error("Local cache is empty. Enroll at least one person.");
  }

  await ensureModels();
  setStatus("Detecting face and running identification...");

  const threshold = toNumberInRange(elements.thresholdInput.value, 0.1, 0.99, 0.52);
  const { embedding } = await detectAndEmbed();
  const best = bestCosineMatch(embedding, people, threshold);

  if (best.matched) {
    const title = formatPerson(best.person);
    setResult(
      `Matched: ${title} (id: ${best.person.personId}), score=${best.score.toFixed(4)}`,
      "ok"
    );
  } else if (best.person) {
    const title = formatPerson(best.person);
    setResult(
      `No exact match (threshold=${threshold}). Closest: ${title}, score=${best.score.toFixed(4)}`,
      "warn"
    );
  } else {
    setResult("No match found.", "warn");
  }
}

async function detectAndEmbed() {
  const faces = await detectFaces(appState.detectorSession, appState.sourceCanvas, {
    scoreThreshold: 0.7,
    iouThreshold: 0.3,
    maxResults: 5
  });
  const primaryFace = getPrimaryFace(faces);
  renderFrame(faces, primaryFace);

  if (!primaryFace) {
    throw new Error("No face detected in current frame.");
  }

  const embedding = await extractEmbedding(
    appState.recognizerSession,
    appState.sourceCanvas,
    primaryFace,
    { padding: 0.3 }
  );
  return { faces, primaryFace, embedding };
}

async function ensureModels() {
  if (appState.detectorSession && appState.recognizerSession) {
    return;
  }
  setStatus("Loading ONNX models...");

  const [detectorSession, recognizerSession] = await Promise.all([
    appState.detectorSession || createDetectorSession(MODEL_PATHS.detector),
    appState.recognizerSession || createRecognizerSession(MODEL_PATHS.recognizer)
  ]);

  appState.detectorSession = detectorSession;
  appState.recognizerSession = recognizerSession;
  setStatus("Models loaded.");
}

function setSourceFrom(mediaElement) {
  const size = getMediaSize(mediaElement);
  const maxSide = 1280;
  const scale = Math.min(1, maxSide / Math.max(size.width, size.height));
  const width = Math.max(1, Math.round(size.width * scale));
  const height = Math.max(1, Math.round(size.height * scale));

  appState.sourceCanvas.width = width;
  appState.sourceCanvas.height = height;
  appState.sourceCtx.clearRect(0, 0, width, height);
  appState.sourceCtx.drawImage(mediaElement, 0, 0, width, height);
}

function renderFrame(faces = [], primaryFace = null) {
  if (!hasSourceFrame()) {
    return;
  }
  const canvas = elements.frameCanvas;
  const ctx = canvas.getContext("2d");
  canvas.width = appState.sourceCanvas.width;
  canvas.height = appState.sourceCanvas.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(appState.sourceCanvas, 0, 0);

  for (const face of faces) {
    const isPrimary = primaryFace && face === primaryFace;
    ctx.strokeStyle = isPrimary ? "#00b873" : "#0f62fe";
    ctx.lineWidth = isPrimary ? 3 : 2;
    ctx.strokeRect(face.x, face.y, face.width, face.height);
    ctx.fillStyle = isPrimary ? "#00b873" : "#0f62fe";
    ctx.font = "14px sans-serif";
    ctx.fillText(
      `${isPrimary ? "Primary" : "Face"} ${face.score.toFixed(2)}`,
      face.x + 4,
      Math.max(16, face.y - 6)
    );
  }
}

function hasSourceFrame() {
  return appState.sourceCanvas.width > 0 && appState.sourceCanvas.height > 0;
}

function renderPeopleList() {
  const people = listPeople();
  if (!people.length) {
    elements.peopleList.innerHTML =
      "<p class='muted'>No embeddings in cache. Enroll a person to begin.</p>";
    return;
  }

  elements.peopleList.innerHTML = "";
  for (const person of people) {
    const row = document.createElement("div");
    row.className = "person";

    const info = document.createElement("div");
    const title = document.createElement("div");
    title.className = "person-title";
    title.textContent = `${formatPerson(person)} [${person.personId}]`;

    const subtitle = document.createElement("div");
    subtitle.className = "person-subtitle";
    subtitle.textContent = `Updated: ${new Date(person.updatedAt).toLocaleString()}`;
    info.appendChild(title);
    info.appendChild(subtitle);

    const removeButton = document.createElement("button");
    removeButton.textContent = "Delete";
    removeButton.addEventListener("click", () => {
      removePerson(person.personId);
      renderPeopleList();
      setStatus(`Deleted ${person.personId} from local cache.`);
    });

    row.appendChild(info);
    row.appendChild(removeButton);
    elements.peopleList.appendChild(row);
  }
}

function clearCache() {
  clearPeopleCache();
  renderPeopleList();
  setResult("No identification run yet.");
  setStatus("Local cache cleared.");
}

function setStatus(text) {
  elements.status.textContent = text;
}

function setResult(text, kind = "") {
  elements.result.classList.remove("ok", "warn");
  if (kind) {
    elements.result.classList.add(kind);
  }
  elements.result.textContent = text;
}

function formatPerson(person) {
  const name = [person.firstName, person.lastName].filter(Boolean).join(" ").trim();
  return name || "Unnamed Person";
}

function getMediaSize(media) {
  if ("videoWidth" in media && media.videoWidth > 0) {
    return { width: media.videoWidth, height: media.videoHeight };
  }
  if ("naturalWidth" in media && media.naturalWidth > 0) {
    return { width: media.naturalWidth, height: media.naturalHeight };
  }
  return { width: media.width, height: media.height };
}

function toNumberInRange(value, min, max, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, parsed));
}

function loadImageElement(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load selected image."));
    img.src = url;
  });
}

init();
