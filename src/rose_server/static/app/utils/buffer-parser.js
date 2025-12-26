export function normalizeTokens(tokens) {
  const normalized = [];

  for (const token of tokens) {
    if (token.type === "text") {
      if (normalized.length > 0 && normalized[normalized.length - 1].type === "text") {
        const prev = normalized[normalized.length - 1];
        prev.value += token.value;
      } else {
        normalized.push({ type: "text", value: token.value });
      }
    } else {
      normalized.push(token);
    }
  }

  if (normalized.length === 0 || normalized[normalized.length - 1].type !== "text") {
    normalized.push({ type: "text", value: "" });
  }

  return normalized;
}

export function serializeTokens(tokens) {
  const parts = [];

  for (const token of tokens) {
    if (token.type === "text") {
      if (token.value) parts.push(token.value);
    }
  }

  return parts.join("");
}

export function parseQuery(queryString, lensTokens = []) {
  const tokens = [];

  for (const lensToken of lensTokens) {
    tokens.push({
      type: "lens",
      lensId: lensToken.lensId,
      atName: lensToken.atName
    });
  }

  if (queryString) {
    tokens.push({ type: "text", value: queryString });
  }

  return normalizeTokens(tokens);
}
