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
  detector: new URL("./models/version-RFB-320.onnx", window.location.href).toString(),
  recognizer: new URL("./models/w600k_mbf.onnx", window.location.href).toString()
};
const ORT_CDN_ASSETS = [
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
const AUTO_WORKER_URL = new URL("./workers/auto-infer.worker.js", import.meta.url);

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
ort.env.wasm.simd = true;

const elements = {
  video: document.getElementById("camera"),
  viewer: document.querySelector(".viewer"),
  frameCanvas: document.getElementById("frameCanvas"),
  status: document.getElementById("status"),
  toggleRecognitionBtn: document.getElementById("toggleRecognitionBtn"),
  addPersonBtn: document.getElementById("addPersonBtn"),
  openMoreBtn: document.getElementById("openMoreBtn"),
  capturePanel: document.getElementById("capturePanel"),
  closeCapturePanelBtn: document.getElementById("closeCapturePanelBtn"),
  actionsModal: document.getElementById("actionsModal"),
  settingsModal: document.getElementById("settingsModal"),
  enrollModal: document.getElementById("enrollModal"),
  openSettingsBtn: document.getElementById("openSettingsBtn"),
  openCacheBtn: document.getElementById("openCacheBtn"),
  cacheModal: document.getElementById("cacheModal"),
  autoThresholdInput: document.getElementById("autoThresholdInput"),
  autoIntervalInput: document.getElementById("autoIntervalInput"),
  autoConfirmInput: document.getElementById("autoConfirmInput"),
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
  worker: {
    instance: null,
    ready: false,
    nextRequestId: 1,
    pending: new Map()
  },
  busy: false,
  peopleCache: [],
  sourceCanvas: document.createElement("canvas"),
  sourceCtx: null,
  auto: {
    running: false,
    runId: 0,
    useWorker: false,
    timerId: null,
    intervalMs: 320,
    confirmFrames: 3,
    detectIntervalMs: 960,
    lastDetectAt: 0,
    lastPrimaryFace: null,
    pendingPersonId: null,
    pendingCount: 0,
    lastLatencyMs: null
  }
};
appState.sourceCtx = appState.sourceCanvas.getContext("2d");

async function init() {
  bindEvents();
  await registerServiceWorker();
  scheduleOfflineWarmup();
  await refreshPeopleCache();
  await renderPeopleList();
  setAutoButtons();
  hideCapturePanel();
  setStatus("Нажмите «Запустить распознавание» или «Добавить человека».");
  setAutoResult("Автораспознавание не запущено.");
  setEnrollResult("Добавление ещё не запускалось.");
  updateAutoMetrics(null);
}

function bindEvents() {
  elements.toggleRecognitionBtn.addEventListener("click", () => runSafe(toggleRecognition));
  elements.addPersonBtn.addEventListener("click", () => runSafe(beginEnrollmentCapture));
  elements.openMoreBtn.addEventListener("click", () => openModal(elements.actionsModal));
  elements.closeCapturePanelBtn.addEventListener("click", hideCapturePanel);
  elements.openSettingsBtn.addEventListener("click", () => {
    closeModal(elements.actionsModal);
    openModal(elements.settingsModal);
  });
  elements.openCacheBtn.addEventListener("click", () =>
    runSafe(async () => {
      closeModal(elements.actionsModal);
      await refreshPeopleCache();
      await renderPeopleList();
      openModal(elements.cacheModal);
    })
  );
  elements.captureEnrollBtn.addEventListener("click", () => runSafe(captureForEnrollment));
  elements.enrollImageInput.addEventListener("change", (event) =>
    runSafe(() => loadEnrollmentImage(event, { openEnrollModal: true }))
  );
  elements.enrollBtn.addEventListener("click", () => runSafe(enrollFace));
  elements.clearCacheBtn.addEventListener("click", () => runSafe(clearCache));
  bindModalEvents();
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
    const message = error?.message || "Операция не выполнена.";
    setStatus(message);
  } finally {
    appState.busy = false;
    toggleBusyControls(false);
    setAutoButtons();
  }
}

function toggleBusyControls(disabled) {
  elements.toggleRecognitionBtn.disabled = disabled;
  elements.addPersonBtn.disabled = disabled || appState.auto.running;
  elements.openMoreBtn.disabled = disabled;
  elements.closeCapturePanelBtn.disabled = disabled;
  elements.openSettingsBtn.disabled = disabled;
  elements.openCacheBtn.disabled = disabled;
  elements.captureEnrollBtn.disabled = disabled;
  elements.enrollBtn.disabled = disabled;
  elements.clearCacheBtn.disabled = disabled;
  elements.enrollImageInput.disabled = disabled;
}

async function toggleRecognition() {
  if (appState.auto.running) {
    stopAutoIdentification("Автораспознавание остановлено.");
    return;
  }

  if (!appState.stream || !elements.video.videoWidth) {
    await startCamera();
  }
  await startAutoIdentification();
}

async function beginEnrollmentCapture() {
  if (appState.auto.running) {
    stopAutoIdentification("Автораспознавание приостановлено для добавления.");
  }
  if (!appState.stream || !elements.video.videoWidth) {
    await startCamera();
  }
  showCapturePanel();
  setStatus("Наведите лицо в кадр и нажмите «Снять текущий кадр».");
}

function showCapturePanel() {
  elements.capturePanel.classList.remove("hidden");
}

function hideCapturePanel() {
  elements.capturePanel.classList.add("hidden");
}

async function startCamera() {
  if (appState.stream) {
    setStatus("Камера уже активна.");
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("API камеры недоступен в этом браузере.");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
    audio: false
  });
  appState.stream = stream;
  elements.video.srcObject = stream;
  await elements.video.play();
  setStatus("Камера запущена.");
}

function stopCamera() {
  stopAutoIdentification("");

  if (!appState.stream) {
    setStatus("Камера уже остановлена.");
    return;
  }

  for (const track of appState.stream.getTracks()) {
    track.stop();
  }
  appState.stream = null;
  elements.video.srcObject = null;
  setStatus("Камера остановлена.");
}

async function captureForEnrollment() {
  if (!appState.stream || !elements.video.videoWidth) {
    throw new Error("Сначала запустите камеру.");
  }
  setSourceFrom(elements.video);
  renderFrame([], null, { drawSource: true });
  setEnrollResult("Кадр для добавления сохранен.");
  setStatus("Текущий кадр с камеры захвачен.");
  hideCapturePanel();
  openModal(elements.enrollModal);
}

async function loadEnrollmentImage(event, options = {}) {
  const openEnrollModal = options.openEnrollModal ?? false;
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }

  const imageUrl = URL.createObjectURL(file);
  try {
    const image = await loadImageElement(imageUrl);
    setSourceFrom(image);
    renderFrame([], null, { drawSource: true });
    setEnrollResult(`Фото загружено: ${file.name}`);
    setStatus(`Источник для добавления обновлен: ${file.name}.`);
    if (openEnrollModal) {
      hideCapturePanel();
      openModal(elements.enrollModal);
    }
  } finally {
    URL.revokeObjectURL(imageUrl);
    elements.enrollImageInput.value = "";
  }
}

async function enrollFace() {
  if (appState.auto.running) {
    stopAutoIdentification("Автораспознавание приостановлено для добавления.");
  }

  const personId = elements.personIdInput.value.trim();
  const firstName = elements.firstNameInput.value.trim();
  const lastName = elements.lastNameInput.value.trim();

  if (!personId) {
    throw new Error("Нужно указать ID человека.");
  }

  if (!hasSourceFrame()) {
    if (appState.stream && elements.video.videoWidth) {
      setSourceFrom(elements.video);
      renderFrame([], null, { drawSource: true });
    } else {
      throw new Error("Сначала снимите кадр или загрузите фото.");
    }
  }

  await ensureModels();
  setStatus("Определяем лицо и создаем эмбеддинг...");
  setEnrollResult("Обрабатываем добавление...");

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
    `Сохранён сэмпл №${person.sampleCount} для "${personId}".`,
    "ok"
  );
  setStatus(
    `Сэмпл №${person.sampleCount} для "${personId}" сохранен в IndexedDB.`
  );
  closeModal(elements.enrollModal);
}

async function startAutoIdentification() {
  if (appState.auto.running) {
    setStatus("Автораспознавание уже запущено.");
    return;
  }

  if (!appState.stream || !elements.video.videoWidth) {
    throw new Error("Сначала запустите камеру.");
  }

  await refreshPeopleCache();
  if (!appState.peopleCache.length) {
    throw new Error("Кэш пуст. Сначала добавьте хотя бы одного человека.");
  }

  appState.auto.useWorker = await ensureAutoWorker();
  if (!appState.auto.useWorker) {
    await ensureModels();
    setStatus("Воркер недоступен, инференс выполняется в основном потоке.");
  } else {
    await workerCall("resetTracking", {});
  }

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
  appState.auto.detectIntervalMs = Math.max(
    900,
    Math.round(appState.auto.intervalMs * 3)
  );
  hideCapturePanel();
  closeModal(elements.enrollModal);
  resetAutoTracking();
  resetAutoConfirmation();
  appState.auto.runId += 1;
  appState.auto.running = true;
  setAutoButtons();
  setAutoResult("Распознавание запущено.");
  setStatus(
    appState.auto.useWorker
      ? "Автораспознавание выполняется в воркере."
      : "Автораспознавание запущено."
  );
  void autoIdentifyTick(appState.auto.runId);
}

function stopAutoIdentification(statusText = "Автораспознавание остановлено.") {
  if (appState.auto.timerId) {
    window.clearTimeout(appState.auto.timerId);
    appState.auto.timerId = null;
  }
  appState.auto.runId += 1;
  appState.auto.running = false;
  appState.auto.useWorker = false;
  resetAutoTracking();
  resetAutoConfirmation();
  teardownAutoWorker();
  clearFrameOverlay();
  setAutoButtons();
  if (statusText) {
    setStatus(statusText);
  }
}

async function autoIdentifyTick(runId) {
  if (!appState.auto.running || runId !== appState.auto.runId) {
    return;
  }

  let runDetectionThisTick = false;
  let latencyMs = null;
  const startedAt = performance.now();
  try {
    if (!appState.stream || !elements.video.videoWidth) {
      throw new Error("Камера недоступна.");
    }
    if (!appState.peopleCache.length) {
      throw new Error("Кэш пуст. Добавьте хотя бы одного человека.");
    }

    const baseThreshold = toNumberInRange(
      elements.autoThresholdInput.value,
      0.1,
      0.99,
      0.55
    );
    const minGap = appState.peopleCache.length > 1 ? 0.03 : 0;

    if (appState.auto.useWorker) {
      const frameResult = await processAutoFrameInWorker({
        baseThreshold,
        minGap,
        maxSide: 480,
        detectIntervalMs: appState.auto.detectIntervalMs
      });
      latencyMs = frameResult.latencyMs;
      runDetectionThisTick = !!frameResult.ranDetection;
      renderAutoOverlay(
        frameResult.frameWidth,
        frameResult.frameHeight,
        frameResult.faces || [],
        frameResult.primaryFace,
        frameResult.best || null
      );

      if (frameResult.noFace || !frameResult.best) {
        resetAutoConfirmation();
        setAutoResult("На текущем кадре лицо не найдено.", "warn");
      } else {
        handleAutoBestMatch(
          frameResult.best,
          frameResult.threshold,
          frameResult.minGap
        );
      }
    } else {
      setSourceFrom(elements.video, { maxSide: 480 });
      const threshold = getRecommendedThreshold(appState.peopleCache.length, baseThreshold);
      runDetectionThisTick =
        !appState.auto.lastPrimaryFace ||
        startedAt - appState.auto.lastDetectAt >= appState.auto.detectIntervalMs;
      const { embedding, primaryFace, faces } = await detectAndEmbed({
        overlayOnly: true,
        runDetection: runDetectionThisTick,
        reuseFaceBox: appState.auto.lastPrimaryFace
      });
      if (runDetectionThisTick) {
        appState.auto.lastDetectAt = startedAt;
      }
      appState.auto.lastPrimaryFace = primaryFace ? cloneFaceBox(primaryFace) : null;
      const best = bestCosineMatch(embedding, appState.peopleCache, threshold, { minGap });
      handleAutoBestMatch(best, threshold, minGap);
      renderFrame(faces, primaryFace, { drawSource: false, best });
    }
  } catch (error) {
    if (runId !== appState.auto.runId) {
      return;
    }
    resetAutoTracking();
    resetAutoConfirmation();
    const message = error?.message || "Ошибка автораспознавания.";
    let switchedFromWorker = false;

    if (appState.auto.useWorker) {
      switchedFromWorker = true;
      appState.auto.useWorker = false;
      appState.worker.ready = false;
      await ensureModels();
      setStatus("Воркер завершился с ошибкой, переключено на основной поток.");
      setAutoResult("Воркер недоступен, используем инференс в основном потоке.", "warn");
    }

    if (!switchedFromWorker) {
      setAutoResult(`Ошибка автораспознавания: ${message}`, "warn");
      setStatus(message);
    }

    if (
      message.includes("Камера недоступна") ||
      message.includes("Кэш пуст")
    ) {
      stopAutoIdentification("Автораспознавание остановлено.");
      return;
    }
  }

  const finalLatencyMs = latencyMs ?? (performance.now() - startedAt);
  appState.auto.lastLatencyMs = finalLatencyMs;
  updateAutoMetrics(finalLatencyMs, runDetectionThisTick);

  if (!appState.auto.running || runId !== appState.auto.runId) {
    return;
  }

  // Fixed gap between checks keeps camera preview smooth and avoids CPU spikes.
  const waitMs = appState.auto.intervalMs;
  appState.auto.timerId = window.setTimeout(() => {
    void autoIdentifyTick(runId);
  }, waitMs);
}

function handleAutoBestMatch(best, threshold, minGap) {
  if (!best?.person) {
    resetAutoConfirmation();
    setAutoResult("Совпадений не найдено.", "warn");
    return;
  }

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
        `Совпадение: ${title} [${best.person.personId}] • score ${best.score.toFixed(3)} • ${confirmProgress}`,
        "ok"
      );
    } else {
      setAutoResult(
        `Кандидат: ${title} [${best.person.personId}] • score ${best.score.toFixed(3)} • ${confirmProgress}`,
        "warn"
      );
    }
    return;
  }

  resetAutoConfirmation();
  const title = formatPerson(best.person);
  const gapText = best.gap == null ? "н/д" : best.gap.toFixed(4);
  setAutoResult(
    `Нет точного совпадения • ${title} • score ${best.score.toFixed(3)} • gap ${gapText}`,
    "warn"
  );
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
    throw new Error("На текущем кадре лицо не найдено.");
  }

  const embedding = await extractEmbedding(
    appState.recognizerSession,
    appState.sourceCanvas,
    primaryFace,
    { padding: 0.3 }
  );

  return { embedding, primaryFace, faces };
}

async function ensureModels() {
  if (appState.detectorSession && appState.recognizerSession) {
    return;
  }
  setStatus("Загружаем ONNX-модели...");

  const [detectorSession, recognizerSession] = await Promise.all([
    appState.detectorSession || createDetectorSession(MODEL_PATHS.detector),
    appState.recognizerSession || createRecognizerSession(MODEL_PATHS.recognizer)
  ]);

  appState.detectorSession = detectorSession;
  appState.recognizerSession = recognizerSession;
  setStatus("Модели загружены.");
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
  const best = options.best ?? null;
  const sourceWidth = appState.sourceCanvas.width;
  const sourceHeight = appState.sourceCanvas.height;
  const overlay = prepareOverlayCanvas();
  if (!overlay) {
    return;
  }

  const { ctx, width, height } = overlay;
  ctx.clearRect(0, 0, width, height);
  if (drawSource) {
    ctx.drawImage(appState.sourceCanvas, 0, 0, width, height);
  }
  drawFaceBoxes(ctx, faces, primaryFace, best, {
    sourceWidth,
    sourceHeight,
    canvasWidth: width,
    canvasHeight: height
  });
}

function renderAutoOverlay(frameWidth, frameHeight, faces = [], primaryFace = null, best = null) {
  const overlay = prepareOverlayCanvas();
  if (!overlay) {
    return;
  }

  const { ctx, width, height } = overlay;
  ctx.clearRect(0, 0, width, height);
  drawFaceBoxes(ctx, faces, primaryFace, best, {
    sourceWidth: frameWidth,
    sourceHeight: frameHeight,
    canvasWidth: width,
    canvasHeight: height
  });
}

function drawFaceBoxes(ctx, faces, primaryFace, best = null, layoutOptions = null) {
  const layout = createOverlayLayout(layoutOptions);
  for (const face of faces) {
    const isPrimary = isSameFaceBox(face, primaryFace);
    const person = isPrimary ? best?.person : null;
    const identity = person ? `${formatPerson(person)} [${person.personId}]` : "";
    const faceForOverlay = mapFaceBoxToOverlay(face, layout);
    if (!faceForOverlay) {
      continue;
    }
    drawPremiumFaceBox(ctx, faceForOverlay, {
      isPrimary,
      identity,
      matchState: isPrimary && best?.person ? (best.matched ? "MATCH" : "CANDIDATE") : "",
      matchScore: isPrimary && Number.isFinite(best?.score) ? best.score : null
    });
  }
}

function isSameFaceBox(face, primaryFace) {
  if (!face || !primaryFace) {
    return false;
  }
  return (
    Math.abs(face.x - primaryFace.x) < 1 &&
    Math.abs(face.y - primaryFace.y) < 1 &&
    Math.abs(face.width - primaryFace.width) < 1 &&
    Math.abs(face.height - primaryFace.height) < 1
  );
}

function drawPremiumFaceBox(ctx, face, options = {}) {
  const isPrimary = options.isPrimary ?? false;
  const identity = options.identity ?? "";
  const matchState = options.matchState ?? "";
  const matchScore = options.matchScore;
  const x = face.x;
  const y = face.y;
  const w = face.width;
  const h = face.height;
  const radius = Math.max(6, Math.min(16, Math.round(Math.min(w, h) * 0.09)));
  const corner = Math.max(10, Math.min(30, Math.round(Math.min(w, h) * 0.24)));
  const mainColor = isPrimary ? "#22d39b" : "#4b66ff";
  const glowColor = isPrimary ? "rgba(34, 211, 155, 0.45)" : "rgba(75, 102, 255, 0.45)";
  const detectionScore = Number.isFinite(face.score) ? face.score : null;
  const detectionScoreText = detectionScore == null ? "--" : detectionScore.toFixed(2);
  const matchScoreText = Number.isFinite(matchScore) ? matchScore.toFixed(2) : null;
  const titleText = matchState
    ? `${matchState} ${matchScoreText ?? detectionScoreText}`
    : `${isPrimary ? "ACTIVE FACE" : "FACE"} ${detectionScoreText}`;

  ctx.save();

  // Soft halo around the frame to match premium dark style.
  ctx.shadowColor = glowColor;
  ctx.shadowBlur = isPrimary ? 26 : 18;
  ctx.lineWidth = isPrimary ? 3 : 2;
  ctx.strokeStyle = mainColor;
  drawRoundedRectPath(ctx, x, y, w, h, radius);
  ctx.stroke();

  ctx.shadowBlur = 0;
  drawCornerAccents(ctx, x, y, w, h, corner, mainColor, isPrimary ? 3 : 2);
  drawFaceLabel(ctx, x, y, w, titleText, identity, isPrimary, mainColor);

  ctx.restore();
}

function drawCornerAccents(ctx, x, y, w, h, cornerSize, color, lineWidth) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();

  ctx.moveTo(x, y + cornerSize);
  ctx.lineTo(x, y);
  ctx.lineTo(x + cornerSize, y);

  ctx.moveTo(x + w - cornerSize, y);
  ctx.lineTo(x + w, y);
  ctx.lineTo(x + w, y + cornerSize);

  ctx.moveTo(x + w, y + h - cornerSize);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x + w - cornerSize, y + h);

  ctx.moveTo(x + cornerSize, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + h - cornerSize);

  ctx.stroke();
}

function drawFaceLabel(ctx, x, y, width, title, subtitle, isPrimary, mainColor) {
  const titleSize = Math.max(11, Math.min(15, Math.round(width * 0.07)));
  const subtitleSize = Math.max(10, titleSize - 2);
  const hasSubtitle = Boolean(subtitle);
  ctx.font = `700 ${titleSize}px "Space Grotesk", sans-serif`;
  const titleW = ctx.measureText(title).width;
  let subtitleW = 0;
  if (hasSubtitle) {
    ctx.font = `600 ${subtitleSize}px "Space Grotesk", sans-serif`;
    subtitleW = ctx.measureText(subtitle).width;
  }
  const paddingX = 10;
  const paddingY = 6;
  const lineGap = hasSubtitle ? 3 : 0;
  const labelW = Math.max(titleW, subtitleW) + paddingX * 2;
  const labelH = titleSize + (hasSubtitle ? subtitleSize + lineGap : 0) + paddingY * 2;
  const labelX = x + 6;
  const labelY = Math.max(8, y - labelH - 8);
  const labelR = Math.min(10, Math.round(labelH / 2));

  const gradient = ctx.createLinearGradient(labelX, labelY, labelX + labelW, labelY);
  if (isPrimary) {
    gradient.addColorStop(0, "rgba(10, 72, 56, 0.92)");
    gradient.addColorStop(1, "rgba(14, 120, 92, 0.92)");
  } else {
    gradient.addColorStop(0, "rgba(20, 36, 118, 0.92)");
    gradient.addColorStop(1, "rgba(28, 63, 212, 0.92)");
  }

  ctx.fillStyle = gradient;
  drawRoundedRectPath(ctx, labelX, labelY, labelW, labelH, labelR);
  ctx.fill();

  ctx.lineWidth = 1;
  ctx.strokeStyle = mainColor;
  drawRoundedRectPath(ctx, labelX, labelY, labelW, labelH, labelR);
  ctx.stroke();

  ctx.textBaseline = "top";
  ctx.fillStyle = "#f6f9ff";
  ctx.font = `700 ${titleSize}px "Space Grotesk", sans-serif`;
  ctx.fillText(title, labelX + paddingX, labelY + paddingY);

  if (hasSubtitle) {
    ctx.fillStyle = "rgba(234, 241, 255, 0.92)";
    ctx.font = `600 ${subtitleSize}px "Space Grotesk", sans-serif`;
    ctx.fillText(subtitle, labelX + paddingX, labelY + paddingY + titleSize + lineGap);
  }
}

function drawRoundedRectPath(ctx, x, y, w, h, r) {
  const radius = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.arcTo(x + w, y, x + w, y + radius, radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.arcTo(x + w, y + h, x + w - radius, y + h, radius);
  ctx.lineTo(x + radius, y + h);
  ctx.arcTo(x, y + h, x, y + h - radius, radius);
  ctx.lineTo(x, y + radius);
  ctx.arcTo(x, y, x + radius, y, radius);
  ctx.closePath();
}

function prepareOverlayCanvas() {
  const canvas = elements.frameCanvas;
  const ctx = canvas.getContext("2d");
  const rect = elements.viewer?.getBoundingClientRect();
  const cssWidth = Math.max(1, Math.round(rect?.width || elements.video.clientWidth || 1));
  const cssHeight = Math.max(1, Math.round(rect?.height || elements.video.clientHeight || 1));
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(cssWidth * dpr));
  const height = Math.max(1, Math.round(cssHeight * dpr));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  return { canvas, ctx, width, height, dpr };
}

function createOverlayLayout(options = {}) {
  const sourceWidth = options?.sourceWidth ?? 0;
  const sourceHeight = options?.sourceHeight ?? 0;
  const canvasWidth = options?.canvasWidth ?? 0;
  const canvasHeight = options?.canvasHeight ?? 0;
  if (!sourceWidth || !sourceHeight || !canvasWidth || !canvasHeight) {
    return null;
  }

  const scale = Math.max(canvasWidth / sourceWidth, canvasHeight / sourceHeight);
  const drawnWidth = sourceWidth * scale;
  const drawnHeight = sourceHeight * scale;

  return {
    scale,
    offsetX: (canvasWidth - drawnWidth) / 2,
    offsetY: (canvasHeight - drawnHeight) / 2
  };
}

function mapFaceBoxToOverlay(face, layout) {
  if (!face) {
    return null;
  }
  if (!layout) {
    return face;
  }

  return {
    x: layout.offsetX + face.x * layout.scale,
    y: layout.offsetY + face.y * layout.scale,
    width: face.width * layout.scale,
    height: face.height * layout.scale,
    score: face.score
  };
}

function clearFrameOverlay() {
  const canvas = elements.frameCanvas;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function hasSourceFrame() {
  return appState.sourceCanvas.width > 0 && appState.sourceCanvas.height > 0;
}

async function refreshPeopleCache() {
  appState.peopleCache = await listPeople();
  if (appState.worker.ready) {
    await syncPeopleToWorker();
  }
}

async function renderPeopleList() {
  const people = appState.peopleCache;
  if (!people.length) {
    elements.peopleList.innerHTML =
      "<p class='muted'>В кэше нет эмбеддингов. Добавьте человека, чтобы начать.</p>";
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
    subtitle.textContent = `Сэмплы: ${person.sampleCount} | Обновлено: ${new Date(person.updatedAt).toLocaleString()}`;
    info.appendChild(title);
    info.appendChild(subtitle);

    const removeButton = document.createElement("button");
    removeButton.textContent = "Удалить";
    removeButton.addEventListener("click", () =>
      runSafe(async () => {
        await removePerson(person.personId);
        await refreshPeopleCache();
        await renderPeopleList();
        if (!appState.peopleCache.length) {
          stopAutoIdentification("Автораспознавание остановлено: кэш стал пустым.");
        }
        setStatus(`${person.personId} удален из кэша.`);
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
  stopAutoIdentification("Автораспознавание остановлено: кэш очищен.");
  setAutoResult("Автораспознавание не запущено.");
  setEnrollResult("Добавление ещё не запускалось.");
  setStatus("Кэш очищен.");
  closeModal(elements.actionsModal);
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
  element.title = text;
}

function setAutoButtons() {
  const running = appState.auto.running;
  elements.toggleRecognitionBtn.textContent = running
    ? "Остановить распознавание"
    : "Запустить распознавание";
  elements.toggleRecognitionBtn.classList.toggle("danger", running);
  elements.addPersonBtn.classList.toggle("hidden", running);
  if (running) {
    hideCapturePanel();
  }
}

function bindModalEvents() {
  for (const closer of document.querySelectorAll("[data-close-modal]")) {
    closer.addEventListener("click", () => {
      const modalId = closer.getAttribute("data-close-modal");
      if (modalId) {
        closeModalById(modalId);
      }
    });
  }

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") {
      return;
    }
    closeModal(elements.actionsModal);
    closeModal(elements.settingsModal);
    closeModal(elements.enrollModal);
    closeModal(elements.cacheModal);
  });
}

function openModal(modal) {
  if (!modal) {
    return;
  }
  modal.classList.remove("hidden");
  modal.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
}

function closeModal(modal) {
  if (!modal || modal.classList.contains("hidden")) {
    return;
  }
  modal.classList.add("hidden");
  modal.setAttribute("aria-hidden", "true");
  const anyOpen = Array.from(document.querySelectorAll(".modal")).some(
    (item) => !item.classList.contains("hidden")
  );
  if (!anyOpen) {
    document.body.classList.remove("modal-open");
  }
}

function closeModalById(modalId) {
  if (modalId === "actionsModal") {
    closeModal(elements.actionsModal);
    return;
  }
  if (modalId === "settingsModal") {
    closeModal(elements.settingsModal);
    return;
  }
  if (modalId === "enrollModal") {
    closeModal(elements.enrollModal);
    return;
  }
  if (modalId === "cacheModal") {
    closeModal(elements.cacheModal);
  }
}

async function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) {
    return;
  }

  try {
    const swUrl = new URL("./service-worker.js", window.location.href).toString();
    const registration = await navigator.serviceWorker.register(swUrl, { scope: "./" });

    if (registration.waiting) {
      registration.waiting.postMessage({ type: "SKIP_WAITING" });
    }

    registration.addEventListener("updatefound", () => {
      const installingWorker = registration.installing;
      if (!installingWorker) {
        return;
      }
      installingWorker.addEventListener("statechange", () => {
        if (installingWorker.state === "installed" && navigator.serviceWorker.controller) {
          installingWorker.postMessage({ type: "SKIP_WAITING" });
        }
      });
    });
  } catch (error) {
    const message = error?.message || "Не удалось зарегистрировать Service Worker.";
    setStatus(`${message} Работа продолжится онлайн.`);
  }
}

function scheduleOfflineWarmup() {
  const startWarmup = () => {
    void warmupOfflineAssets();
  };

  if (typeof window.requestIdleCallback === "function") {
    window.requestIdleCallback(() => startWarmup(), { timeout: 2500 });
    return;
  }

  window.setTimeout(startWarmup, 1500);
}

async function warmupOfflineAssets() {
  if (!navigator.onLine) {
    return;
  }
  if (!("serviceWorker" in navigator) || !navigator.serviceWorker.controller) {
    return;
  }

  const urls = [
    MODEL_PATHS.detector,
    MODEL_PATHS.recognizer,
    AUTO_WORKER_URL.toString(),
    ...ORT_CDN_ASSETS
  ];

  await Promise.all(
    urls.map((url) =>
      fetch(url, { cache: "reload" }).catch(() => null)
    )
  );
}

function updateAutoMetrics(latencyMs, ranDetection = false) {
  if (latencyMs == null) {
    elements.autoMetrics.textContent = "Задержка: - мс | FPS инференса: - | Проверок/с: -";
    return;
  }

  const inferenceFps = latencyMs > 0 ? (1000 / latencyMs).toFixed(2) : "-";
  const checksPerSecond = (1000 / (latencyMs + appState.auto.intervalMs)).toFixed(2);
  elements.autoMetrics.textContent =
    `Задержка: ${latencyMs.toFixed(0)} мс | FPS инференса: ${inferenceFps} | Проверок/с: ${checksPerSecond} | Детекция: ${ranDetection ? "да" : "повтор рамки"}`;
}

function resetAutoConfirmation() {
  appState.auto.pendingPersonId = null;
  appState.auto.pendingCount = 0;
}

function resetAutoTracking() {
  appState.auto.lastDetectAt = 0;
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
  return name || "Без имени";
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
    img.onerror = () => reject(new Error("Не удалось загрузить выбранное изображение."));
    img.src = url;
  });
}

function teardownAutoWorker() {
  for (const [, pending] of appState.worker.pending) {
    pending.reject(new Error("Воркер автораспознавания перезапущен."));
  }
  appState.worker.pending.clear();
  appState.worker.ready = false;
  appState.worker.nextRequestId = 1;
  if (appState.worker.instance) {
    appState.worker.instance.terminate();
    appState.worker.instance = null;
  }
}

async function ensureAutoWorker() {
  if (!supportsAutoWorker()) {
    return false;
  }

  if (appState.worker.ready) {
    return true;
  }

  if (!appState.worker.instance) {
    createAutoWorkerInstance();
  }

  try {
    await workerCall("init", { modelPaths: MODEL_PATHS });
    appState.worker.ready = true;
    await syncPeopleToWorker();
    return true;
  } catch {
    teardownAutoWorker();
    return false;
  }
}

function supportsAutoWorker() {
  return (
    typeof Worker !== "undefined" &&
    typeof OffscreenCanvas !== "undefined" &&
    typeof createImageBitmap === "function"
  );
}

function createAutoWorkerInstance() {
  const worker = new Worker(AUTO_WORKER_URL, {
    type: "module"
  });

  worker.onmessage = (event) => {
    const { requestId, ok, payload, error } = event.data || {};
    const pending = appState.worker.pending.get(requestId);
    if (!pending) {
      return;
    }
    appState.worker.pending.delete(requestId);
    if (ok) {
      pending.resolve(payload);
    } else {
      pending.reject(new Error(error || "Ошибка запроса к воркеру."));
    }
  };

  worker.onerror = (event) => {
    const message = event?.message || "Ошибка воркера.";
    for (const [, pending] of appState.worker.pending) {
      pending.reject(new Error(message));
    }
    appState.worker.pending.clear();
    appState.worker.ready = false;
    appState.worker.instance = null;
  };

  appState.worker.instance = worker;
}

function workerCall(type, payload = {}, transfer = []) {
  if (!appState.worker.instance) {
    return Promise.reject(new Error("Воркер не инициализирован."));
  }

  const requestId = appState.worker.nextRequestId++;
  return new Promise((resolve, reject) => {
    appState.worker.pending.set(requestId, { resolve, reject });
    appState.worker.instance.postMessage({ requestId, type, payload }, transfer);
  });
}

async function syncPeopleToWorker() {
  if (!appState.worker.ready) {
    return;
  }

  const payloadPeople = appState.peopleCache.map((person) => ({
    personId: person.personId,
    firstName: person.firstName,
    lastName: person.lastName,
    sampleCount: person.sampleCount,
    embeddings: Array.isArray(person.embeddings) ? person.embeddings : []
  }));
  await workerCall("setPeople", { people: payloadPeople });
}

async function processAutoFrameInWorker({
  baseThreshold,
  minGap,
  maxSide,
  detectIntervalMs
}) {
  const imageBitmap = await createImageBitmap(elements.video);
  return workerCall(
    "processFrame",
    {
      imageBitmap,
      baseThreshold,
      minGap,
      maxSide,
      detectIntervalMs,
      timestamp: performance.now()
    },
    [imageBitmap]
  );
}

init().catch((error) => {
  setStatus(error.message || "Ошибка инициализации.");
});
