export function l2Normalize(vector) {
  let norm = 0;
  for (let i = 0; i < vector.length; i += 1) {
    norm += vector[i] * vector[i];
  }
  norm = Math.sqrt(norm);
  if (!norm) {
    return Float32Array.from(vector);
  }

  const out = new Float32Array(vector.length);
  for (let i = 0; i < vector.length; i += 1) {
    out[i] = vector[i] / norm;
  }
  return out;
}

export function dotProduct(a, b) {
  if (a.length !== b.length) {
    throw new Error("Vector dimensions mismatch.");
  }

  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
}

export function cosineSimilarity(a, b) {
  return dotProduct(a, b);
}

export function bestCosineMatch(queryEmbedding, records, threshold) {
  let best = null;

  for (const person of records) {
    const score = cosineSimilarity(queryEmbedding, person.embedding);
    if (!best || score > best.score) {
      best = { person, score };
    }
  }

  if (!best) {
    return { matched: false, score: null, person: null };
  }

  return {
    matched: best.score >= threshold,
    score: best.score,
    person: best.person
  };
}
