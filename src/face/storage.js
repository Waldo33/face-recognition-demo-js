const STORAGE_KEY = "face-identification-cache-v1";

function parseStorage(payload) {
  if (!payload) {
    return [];
  }

  try {
    const parsed = JSON.parse(payload);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .filter((item) => item && typeof item === "object")
      .filter((item) => typeof item.personId === "string")
      .map((item) => ({
        personId: item.personId,
        firstName: typeof item.firstName === "string" ? item.firstName : "",
        lastName: typeof item.lastName === "string" ? item.lastName : "",
        embedding: Array.isArray(item.embedding) ? item.embedding : [],
        createdAt:
          typeof item.createdAt === "string"
            ? item.createdAt
            : new Date().toISOString(),
        updatedAt:
          typeof item.updatedAt === "string"
            ? item.updatedAt
            : new Date().toISOString()
      }))
      .filter((item) => item.embedding.length > 0);
  } catch {
    return [];
  }
}

function readRaw() {
  return parseStorage(window.localStorage.getItem(STORAGE_KEY));
}

function writeRaw(records) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
}

export function listPeople() {
  return readRaw().map((record) => ({
    ...record,
    embedding: Float32Array.from(record.embedding)
  }));
}

export function upsertPerson({
  personId,
  firstName = "",
  lastName = "",
  embedding
}) {
  if (!personId || !personId.trim()) {
    throw new Error("Person ID is required.");
  }
  if (!embedding || !embedding.length) {
    throw new Error("Embedding is empty.");
  }

  const normalizedId = personId.trim();
  const now = new Date().toISOString();
  const data = readRaw();
  const index = data.findIndex((item) => item.personId === normalizedId);
  const payload = {
    personId: normalizedId,
    firstName: firstName.trim(),
    lastName: lastName.trim(),
    embedding: Array.from(embedding),
    updatedAt: now,
    createdAt: index >= 0 ? data[index].createdAt : now
  };

  if (index >= 0) {
    data[index] = payload;
  } else {
    data.push(payload);
  }

  writeRaw(data);
  return payload;
}

export function removePerson(personId) {
  const normalizedId = personId.trim();
  const data = readRaw().filter((item) => item.personId !== normalizedId);
  writeRaw(data);
}

export function clearPeopleCache() {
  window.localStorage.removeItem(STORAGE_KEY);
}
