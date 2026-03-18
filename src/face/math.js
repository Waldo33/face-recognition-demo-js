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

export function bestCosineMatch(queryEmbedding, records, threshold, options = {}) {
  const minGap = options.minGap ?? 0.04;
  const personScores = [];

  for (const person of records) {
    const embeddings = getPersonEmbeddings(person);
    let bestSample = null;

    for (let sampleIndex = 0; sampleIndex < embeddings.length; sampleIndex += 1) {
      let score;
      try {
        score = cosineSimilarity(queryEmbedding, embeddings[sampleIndex]);
      } catch {
        continue;
      }

      if (!bestSample || score > bestSample.score) {
        bestSample = { score, sampleIndex };
      }
    }

    if (bestSample) {
      personScores.push({
        person,
        score: bestSample.score,
        sampleIndex: bestSample.sampleIndex
      });
    }
  }

  if (!personScores.length) {
    return {
      matched: false,
      score: null,
      person: null,
      sampleIndex: null,
      gap: null
    };
  }

  personScores.sort((a, b) => b.score - a.score);
  const best = personScores[0];
  const second = personScores[1] || null;
  const gap = second ? best.score - second.score : null;
  const gapOk = !second || gap >= minGap;

  return {
    matched: best.score >= threshold && gapOk,
    score: best.score,
    person: best.person,
    sampleIndex: best.sampleIndex,
    gap
  };
}

export function getRecommendedThreshold(personCount, baseThreshold) {
  if (personCount <= 1) {
    return Math.min(0.75, Math.max(0.5, baseThreshold));
  }

  return baseThreshold;
}

function getPersonEmbeddings(person) {
  if (Array.isArray(person.embeddings) && person.embeddings.length > 0) {
    return person.embeddings;
  }
  if (person.embedding) {
    return [person.embedding];
  }
  return [];
}
