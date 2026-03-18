const DB_NAME = "face-identification-db";
const DB_VERSION = 1;
const STORE_NAME = "embeddings";
const MAX_SAMPLES_PER_PERSON = 12;

let dbPromise = null;

function getDb() {
  if (!("indexedDB" in window)) {
    throw new Error("IndexedDB is not supported in this browser.");
  }

  if (dbPromise) {
    return dbPromise;
  }

  dbPromise = new Promise((resolve, reject) => {
    const request = window.indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "personId" });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("Failed to open IndexedDB."));
  });

  return dbPromise;
}

function requestToPromise(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("IndexedDB request failed."));
  });
}

function transactionDone(transaction) {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(transaction.error || new Error("Transaction aborted."));
    transaction.onerror = () =>
      reject(transaction.error || new Error("IndexedDB transaction failed."));
  });
}

function toFloat32Embedding(raw) {
  if (raw instanceof Float32Array) {
    return new Float32Array(raw);
  }
  if (raw instanceof ArrayBuffer) {
    return new Float32Array(raw.slice(0));
  }
  if (ArrayBuffer.isView(raw)) {
    return new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  }
  if (Array.isArray(raw)) {
    return Float32Array.from(raw);
  }
  return new Float32Array();
}

function toDbRecord({
  personId,
  firstName,
  lastName,
  embeddings,
  createdAt,
  updatedAt
}) {
  const buffers = embeddings.map((embedding) => toFloat32Embedding(embedding).buffer.slice(0));
  const latestBuffer = buffers.length ? buffers[buffers.length - 1] : new ArrayBuffer(0);

  return {
    personId,
    firstName,
    lastName,
    embeddingBuffers: buffers,
    // Keep the legacy field for backwards compatibility.
    embeddingBuffer: latestBuffer,
    createdAt,
    updatedAt
  };
}

function fromDbRecord(record) {
  const rawBuffers =
    Array.isArray(record.embeddingBuffers) && record.embeddingBuffers.length
      ? record.embeddingBuffers
      : record.embeddingBuffer
        ? [record.embeddingBuffer]
        : [];
  const embeddings = rawBuffers
    .map((buffer) => toFloat32Embedding(buffer))
    .filter((embedding) => embedding.length > 0);

  return {
    personId: record.personId,
    firstName: typeof record.firstName === "string" ? record.firstName : "",
    lastName: typeof record.lastName === "string" ? record.lastName : "",
    embeddings,
    sampleCount: embeddings.length,
    createdAt:
      typeof record.createdAt === "string"
        ? record.createdAt
        : new Date().toISOString(),
    updatedAt:
      typeof record.updatedAt === "string"
        ? record.updatedAt
        : new Date().toISOString()
  };
}

export async function listPeople() {
  const db = await getDb();
  const tx = db.transaction(STORE_NAME, "readonly");
  const store = tx.objectStore(STORE_NAME);
  const rows = await requestToPromise(store.getAll());
  await transactionDone(tx);

  return rows
    .filter((row) => row && typeof row.personId === "string")
    .map(fromDbRecord)
    .filter((person) => person.sampleCount > 0)
    .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
}

export async function upsertPerson({
  personId,
  firstName = "",
  lastName = "",
  embedding,
  appendSample = true
}) {
  if (!personId || !personId.trim()) {
    throw new Error("Person ID is required.");
  }

  const normalizedId = personId.trim();
  const safeEmbedding = toFloat32Embedding(embedding);
  if (!safeEmbedding.length) {
    throw new Error("Embedding is empty.");
  }

  const db = await getDb();
  const readTx = db.transaction(STORE_NAME, "readonly");
  const existing = await requestToPromise(readTx.objectStore(STORE_NAME).get(normalizedId));
  await transactionDone(readTx);
  const existingPerson = existing ? fromDbRecord(existing) : null;

  const now = new Date().toISOString();
  const existingSamples = existingPerson?.embeddings || [];
  const mergedSamples = appendSample ? [...existingSamples, safeEmbedding] : [safeEmbedding];
  const boundedSamples = mergedSamples.slice(-MAX_SAMPLES_PER_PERSON);

  const record = toDbRecord({
    personId: normalizedId,
    firstName: firstName.trim() || existingPerson?.firstName || "",
    lastName: lastName.trim() || existingPerson?.lastName || "",
    embeddings: boundedSamples,
    createdAt: existing?.createdAt || now,
    updatedAt: now
  });

  const writeTx = db.transaction(STORE_NAME, "readwrite");
  writeTx.objectStore(STORE_NAME).put(record);
  await transactionDone(writeTx);
  return fromDbRecord(record);
}

export async function removePerson(personId) {
  const normalizedId = personId.trim();
  if (!normalizedId) {
    return;
  }

  const db = await getDb();
  const tx = db.transaction(STORE_NAME, "readwrite");
  tx.objectStore(STORE_NAME).delete(normalizedId);
  await transactionDone(tx);
}

export async function clearPeopleCache() {
  const db = await getDb();
  const tx = db.transaction(STORE_NAME, "readwrite");
  tx.objectStore(STORE_NAME).clear();
  await transactionDone(tx);
}
