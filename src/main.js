import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.mjs";
import { bestCosineMatch, getRecommendedThreshold } from "./face/math.js";
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
  status: document.getElementById("status"),
  startCameraBtn: document.getElementById("startCameraBtn"),
  stopCameraBtn: document.getElementById("stopCameraBtn"),
  autoCaseTab: document.getElementById("autoCaseTab"),
  enrollCaseTab: document.getElementById("enrollCaseTab"),
  autoCase: document.getElementById("autoCase"),
  enrollCase: document.getElementById("enrollCase"),
  autoThresholdInput: document.getElementById("autoThresholdInput"),
  autoIntervalInput: document.getElementById("autoIntervalInput"),
  autoConfirmInput: document.getElementById("autoConfirmInput"),
  startAutoBtn: document.getElementById("startAutoBtn"),
  stopAutoBtn: document.getElementById("stopAutoBtn"),
  autoMetrics: document.getElementById("autoMetrics"),
  autoResult: document.getElementById("autoResult"),
  captureEnrollBtn: document.getElementById("captureEnrollBtn"),
  enrollImageInput: document.getElementById("enrollImageInput"),
  personIdInput: document.getElementById("personIdInput"),
  firstNameInput: document.getElementById("firstNameInput"),
  lastNameInput: document.getElementById("lastNameInput"),
  enrollBtn: document.getElementById("enrollBtn"),
  enrollResult: document.getElementById("enrollResult"),
  clearCacheBtn: document.getElementById("clearCacheBtn"),
  peopleList: document.getElementById("peopleList")
};

const appState = {
  stream: null,
  detectorSession: null,
  recognizerSession: null,
  busy: false,
  peopleCache: [],
  sourceCanvas: document.createElement("canvas"),
  sourceCtx: null,
  auto: {
    running: false,
    timerId: null,
    intervalMs: 320,
    confirmFrames: 3,
    detectEveryCycles: 3,
    tickIndex: 0,
    lastPrimaryFace: null,
    pendingPersonId: null,
    pendingCount: 0,
    lastLatencyMs: null
  }
};
appState.sourceCtx = appState.sourceCanvas.getContext("2d");

async function init() {
  bindEvents();
  await refreshPeopleCache();
  await renderPeopleList();
  switchCase("auto");
  setAutoButtons();
  setStatus("Start camera and pick a workflow below.");
  setAutoResult("Auto identification is idle.");
  setEnrollResult("No enrollment run yet.");
  updateAutoMetrics(null);
}

function bindEvents() {
  elements.startCameraBtn.addEventListener("click", () => runSafe(startCamera));
  elements.stopCameraBtn.addEventListener("click", stopCamera);
  elements.autoCaseTab.addEventListener("click", () => switchCase("auto"));
  elements.enrollCaseTab.addEventListener("click", () => switchCase("enroll"));
  elements.startAutoBtn.addEventListener("click", () => runSafe(startAutoIdentification));
  elements.stopAutoBtn.addEventListener("click", () => stopAutoIdentification("Auto identification stopped."));
  elements.captureEnrollBtn.addEventListener("click", () => runSafe(captureForEnrollment));
  elements.enrollImageInput.addEventListener("change", (event) =>
    runSafe(() => loadEnrollmentImage(event))
  );
  elements.enrollBtn.addEventListener("click", () => runSafe(enrollFace));
  elements.clearCacheBtn.addEventListener("click", () => runSafe(clearCache));
}

async function runSafe(task) {
  if (appState.busy) {
    return;
  }
  appState.busy = true;
  toggleBusyControls(true);
  try {
    await task();
  } catch (error) {
    const message = error?.message || "Operation failed.";
    setStatus(message);
  } finally {
    appState.busy = false;
    toggleBusyControls(false);
    setAutoButtons();
  }
}

function toggleBusyControls(disabled) {
  elements.startCameraBtn.disabled = disabled;
  elements.captureEnrollBtn.disabled = disabled;
  elements.enrollBtn.disabled = disabled;
  elements.clearCacheBtn.disabled = disabled;
  elements.enrollImageInput.disabled = disabled;
  elements.startAutoBtn.disabled = disabled || appState.auto.running;
}

function switchCase(caseName) {
  const autoActive = caseName === "auto";
  if (!autoActive && appState.auto.running) {
    stopAutoIdentification("Auto identification paused for enrollment.");
  }
  elements.autoCase.classList.toggle("hidden", !autoActive);
  elements.enrollCase.classList.toggle("hidden", autoActive);
  elements.autoCaseTab.classList.toggle("active", autoActive);
  elements.enrollCaseTab.classList.toggle("active", !autoActive);
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
  setStatus("Camera started.");
}

function stopCamera() {
  stopAutoIdentification("");

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

async function captureForEnrollment() {
  if (!appState.stream || !elements.video.videoWidth) {
    throw new Error("Start camera first.");
  }
  setSourceFrom(elements.video);
  renderFrame([], null, { drawSource: true });
  setEnrollResult("Frame captured for enrollment.");
  setStatus("Captured current camera frame.");
}

async function loadEnrollmentImage(event) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }

  const imageUrl = URL.createObjectURL(file);
  try {
    const image = await loadImageElement(imageUrl);
    setSourceFrom(image);
    renderFrame([], null, { drawSource: true });
    setEnrollResult(`Image loaded: ${file.name}`);
    setStatus(`Enrollment source updated from ${file.name}.`);
  } finally {
    URL.revokeObjectURL(imageUrl);
    elements.enrollImageInput.value = "";
  }
}

async function enrollFace() {
  if (appState.auto.running) {
    stopAutoIdentification("Auto identification paused for enrollment.");
  }

  const personId = elements.personIdInput.value.trim();
  const firstName = elements.firstNameInput.value.trim();
  const lastName = elements.lastNameInput.value.trim();

  if (!personId) {
    throw new Error("Person ID is required.");
  }

  if (!hasSourceFrame()) {
    if (appState.stream && elements.video.videoWidth) {
      setSourceFrom(elements.video);
      renderFrame([], null, { drawSource: true });
    } else {
      throw new Error("Capture a frame or upload an image first.");
    }
  }

  await ensureModels();
  setStatus("Detecting face and creating embedding...");
  setEnrollResult("Processing enrollment...");

  const { embedding } = await detectAndEmbed();
  const person = await upsertPerson({
    personId,
    firstName,
    lastName,
    embedding,
    appendSample: true
  });
  await refreshPeopleCache();
  await renderPeopleList();
  setEnrollResult(
    `Saved sample ${person.sampleCount} for "${personId}".`,
    "ok"
  );
  setStatus(
    `Saved sample ${person.sampleCount} for "${personId}" in IndexedDB.`
  );
}

async function startAutoIdentification() {
  if (appState.auto.running) {
    setStatus("Auto identification is already running.");
    return;
  }

  if (!appState.stream || !elements.video.videoWidth) {
    throw new Error("Start camera first.");
  }

  await refreshPeopleCache();
  if (!appState.peopleCache.length) {
    throw new Error("Cache is empty. Add at least one person first.");
  }

  await ensureModels();

  appState.auto.intervalMs = toNumberInRange(
    elements.autoIntervalInput.value,
    60,
    2000,
    320
  );
  appState.auto.confirmFrames = toNumberInRange(
    elements.autoConfirmInput.value,
    1,
    10,
    3
  );
  resetAutoTracking();
  resetAutoConfirmation();
  appState.auto.running = true;
  setAutoButtons();
  setAutoResult(
    `Auto identification started (interval ${appState.auto.intervalMs}ms, confirm ${appState.auto.confirmFrames} frames).`
  );
  setStatus("Auto identification is running.");
  void autoIdentifyTick();
}

function stopAutoIdentification(statusText = "Auto identification stopped.") {
  if (appState.auto.timerId) {
    window.clearTimeout(appState.auto.timerId);
    appState.auto.timerId = null;
  }
  appState.auto.running = false;
  resetAutoTracking();
  resetAutoConfirmation();
  setAutoButtons();
  if (statusText) {
    setStatus(statusText);
  }
}

async function autoIdentifyTick() {
  if (!appState.auto.running) {
    return;
  }

  const startedAt = performance.now();
  try {
    if (!appState.stream || !elements.video.videoWidth) {
      throw new Error("Camera is not available.");
    }
    if (!appState.peopleCache.length) {
      throw new Error("Cache is empty. Add at least one person.");
    }

    setSourceFrom(elements.video, { maxSide: 480 });
    const baseThreshold = toNumberInRange(
      elements.autoThresholdInput.value,
      0.1,
      0.99,
      0.55
    );
    const threshold = getRecommendedThreshold(appState.peopleCache.length, baseThreshold);
    const minGap = appState.peopleCache.length > 1 ? 0.03 : 0;
    const runDetection =
      appState.auto.tickIndex % appState.auto.detectEveryCycles === 0 ||
      !appState.auto.lastPrimaryFace;
    const { embedding, primaryFace } = await detectAndEmbed({
      overlayOnly: true,
      runDetection,
      reuseFaceBox: appState.auto.lastPrimaryFace
    });
    appState.auto.lastPrimaryFace = primaryFace ? cloneFaceBox(primaryFace) : null;
    appState.auto.tickIndex += 1;
    const best = bestCosineMatch(embedding, appState.peopleCache, threshold, { minGap });

    if (best.matched) {
      const candidateId = best.person.personId;
      if (appState.auto.pendingPersonId === candidateId) {
        appState.auto.pendingCount += 1;
      } else {
        appState.auto.pendingPersonId = candidateId;
        appState.auto.pendingCount = 1;
      }

      const title = formatPerson(best.person);
      const confirmProgress = `${appState.auto.pendingCount}/${appState.auto.confirmFrames}`;
      if (appState.auto.pendingCount >= appState.auto.confirmFrames) {
        setAutoResult(
          `Matched: ${title} (id: ${best.person.personId}, sample ${best.sampleIndex + 1}/${best.person.sampleCount}), score=${best.score.toFixed(4)} | confirmed ${confirmProgress}`,
          "ok"
        );
      } else {
        setAutoResult(
          `Candidate: ${title} (id: ${best.person.personId}), score=${best.score.toFixed(4)} | confirming ${confirmProgress}`,
          "warn"
        );
      }
    } else if (best.person) {
      resetAutoConfirmation();
      const title = formatPerson(best.person);
      const gapText = best.gap == null ? "n/a" : best.gap.toFixed(4);
      setAutoResult(
        `No exact match (threshold=${threshold.toFixed(2)}, gap>=${minGap.toFixed(2)}). Closest: ${title} (sample ${best.sampleIndex + 1}/${best.person.sampleCount}), score=${best.score.toFixed(4)}, gap=${gapText}`,
        "warn"
      );
    } else {
      resetAutoConfirmation();
      setAutoResult("No match found.", "warn");
    }
  } catch (error) {
    resetAutoTracking();
    resetAutoConfirmation();
    const message = error?.message || "Auto identification failed.";
    setAutoResult(`Auto identification failed: ${message}`, "warn");
    setStatus(message);

    if (
      message.includes("Camera is not available") ||
      message.includes("Cache is empty")
    ) {
      stopAutoIdentification("Auto identification stopped.");
      return;
    }
  }

  const latency = performance.now() - startedAt;
  appState.auto.lastLatencyMs = latency;
  updateAutoMetrics(latency);

  if (!appState.auto.running) {
    return;
  }

  // Fixed gap between checks keeps camera preview smooth and avoids CPU spikes.
  const waitMs = appState.auto.intervalMs;
  appState.auto.timerId = window.setTimeout(() => {
    void autoIdentifyTick();
  }, waitMs);
}

async function detectAndEmbed(options = {}) {
  const overlayOnly = options.overlayOnly ?? false;
  const runDetection = options.runDetection ?? true;
  const reuseFaceBox = options.reuseFaceBox ?? null;

  let faces = [];
  let primaryFace = null;

  if (runDetection) {
    faces = await detectFaces(appState.detectorSession, appState.sourceCanvas, {
      scoreThreshold: 0.7,
      iouThreshold: 0.3,
      maxResults: 5
    });
    primaryFace = getPrimaryFace(faces);
  } else if (reuseFaceBox) {
    primaryFace = clampFaceBoxToSource(reuseFaceBox);
    faces = primaryFace ? [primaryFace] : [];
  }

  renderFrame(faces, primaryFace, { drawSource: !overlayOnly });

  if (!primaryFace) {
    throw new Error("No face detected in current frame.");
  }

  const embedding = await extractEmbedding(
    appState.recognizerSession,
    appState.sourceCanvas,
    primaryFace,
    { padding: 0.3 }
  );

  return { embedding, primaryFace };
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

function setSourceFrom(mediaElement, options = {}) {
  const size = getMediaSize(mediaElement);
  const maxSide = options.maxSide ?? 960;
  const scale = Math.min(1, maxSide / Math.max(size.width, size.height));
  const width = Math.max(1, Math.round(size.width * scale));
  const height = Math.max(1, Math.round(size.height * scale));

  if (appState.sourceCanvas.width !== width || appState.sourceCanvas.height !== height) {
    appState.sourceCanvas.width = width;
    appState.sourceCanvas.height = height;
  }

  appState.sourceCtx.clearRect(0, 0, width, height);
  appState.sourceCtx.drawImage(mediaElement, 0, 0, width, height);
}

function renderFrame(faces = [], primaryFace = null, options = {}) {
  if (!hasSourceFrame()) {
    return;
  }
  const drawSource = options.drawSource ?? false;

  const canvas = elements.frameCanvas;
  const ctx = canvas.getContext("2d");
  canvas.width = appState.sourceCanvas.width;
  canvas.height = appState.sourceCanvas.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (drawSource) {
    ctx.drawImage(appState.sourceCanvas, 0, 0);
  }

  for (const face of faces) {
    const isPrimary = primaryFace && face === primaryFace;
    ctx.strokeStyle = isPrimary ? "#00b873" : "#0f62fe";
    ctx.lineWidth = isPrimary ? 3 : 2;
    ctx.strokeRect(face.x, face.y, face.width, face.height);
    ctx.fillStyle = isPrimary ? "#00b873" : "#0f62fe";
    ctx.font = "14px sans-serif";
    const scoreText = Number.isFinite(face.score) ? face.score.toFixed(2) : "--";
    ctx.fillText(
      `${isPrimary ? "Primary" : "Face"} ${scoreText}`,
      face.x + 4,
      Math.max(16, face.y - 6)
    );
  }
}

function hasSourceFrame() {
  return appState.sourceCanvas.width > 0 && appState.sourceCanvas.height > 0;
}

async function refreshPeopleCache() {
  appState.peopleCache = await listPeople();
}

async function renderPeopleList() {
  const people = appState.peopleCache;
  if (!people.length) {
    elements.peopleList.innerHTML =
      "<p class='muted'>No embeddings in cache. Add a person to begin.</p>";
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
    subtitle.textContent = `Samples: ${person.sampleCount} | Updated: ${new Date(person.updatedAt).toLocaleString()}`;
    info.appendChild(title);
    info.appendChild(subtitle);

    const removeButton = document.createElement("button");
    removeButton.textContent = "Delete";
    removeButton.addEventListener("click", () =>
      runSafe(async () => {
        await removePerson(person.personId);
        await refreshPeopleCache();
        await renderPeopleList();
        if (!appState.peopleCache.length) {
          stopAutoIdentification("Auto identification stopped: cache became empty.");
        }
        setStatus(`Deleted ${person.personId} from cache.`);
      })
    );

    row.appendChild(info);
    row.appendChild(removeButton);
    elements.peopleList.appendChild(row);
  }
}

async function clearCache() {
  await clearPeopleCache();
  await refreshPeopleCache();
  await renderPeopleList();
  stopAutoIdentification("Auto identification stopped: cache cleared.");
  setAutoResult("Auto identification is idle.");
  setEnrollResult("No enrollment run yet.");
  setStatus("Cache cleared.");
}

function setStatus(text) {
  elements.status.textContent = text;
}

function setAutoResult(text, kind = "") {
  setMessage(elements.autoResult, text, kind);
}

function setEnrollResult(text, kind = "") {
  setMessage(elements.enrollResult, text, kind);
}

function setMessage(element, text, kind = "") {
  element.classList.remove("ok", "warn");
  if (kind) {
    element.classList.add(kind);
  }
  element.textContent = text;
}

function setAutoButtons() {
  elements.startAutoBtn.disabled = appState.auto.running || appState.busy;
  elements.stopAutoBtn.disabled = !appState.auto.running;
}

function updateAutoMetrics(latencyMs) {
  if (latencyMs == null) {
    elements.autoMetrics.textContent = "Latency: - ms | Inference FPS: - | Checks/s: -";
    return;
  }

  const inferenceFps = latencyMs > 0 ? (1000 / latencyMs).toFixed(2) : "-";
  const checksPerSecond = (1000 / (latencyMs + appState.auto.intervalMs)).toFixed(2);
  elements.autoMetrics.textContent =
    `Latency: ${latencyMs.toFixed(0)} ms | Inference FPS: ${inferenceFps} | Checks/s: ${checksPerSecond}`;
}

function resetAutoConfirmation() {
  appState.auto.pendingPersonId = null;
  appState.auto.pendingCount = 0;
}

function resetAutoTracking() {
  appState.auto.tickIndex = 0;
  appState.auto.lastPrimaryFace = null;
}

function cloneFaceBox(face) {
  return {
    x: face.x,
    y: face.y,
    width: face.width,
    height: face.height,
    score: face.score
  };
}

function clampFaceBoxToSource(face) {
  const width = appState.sourceCanvas.width;
  const height = appState.sourceCanvas.height;
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

init().catch((error) => {
  setStatus(error.message || "Initialization failed.");
});
