const TOKEN_TYPES = {
  TEXT: "text",
  MENTION: "mention",
  HASHTAG: "hashtag",
  COMMAND: "command",
};

const TOKEN_PREFIXES = {
  "@": TOKEN_TYPES.MENTION,
  "#": TOKEN_TYPES.HASHTAG,
  "/": TOKEN_TYPES.COMMAND,
};

const TOKEN_CHAR_REGEX = /[A-Za-z0-9-]/;
const TOKEN_BODY_REGEX = /^[@#/]([A-Za-z0-9-]*)/;

function tokenize(text) {
  if (!text) return [];

  const tokens = [];
  let position = 0;

  while (position < text.length) {
    const remaining = text.slice(position);
    const char = remaining[0];

    const isWordBoundary = position === 0 || !TOKEN_CHAR_REGEX.test(text[position - 1]);

    if (isWordBoundary && TOKEN_PREFIXES[char]) {
      const match = remaining.match(TOKEN_BODY_REGEX);
      if (match) {
        const raw = match[0];
        const value = match[1];
        const type = TOKEN_PREFIXES[char];
        tokens.push({ type, value, raw });
        position += raw.length;
        continue;
      }
    }

    const textMatch = remaining.match(/^[^@#/]+/);
    if (textMatch) {
      const raw = textMatch[0];
      tokens.push({ type: TOKEN_TYPES.TEXT, value: raw, raw });
      position += raw.length;
    } else {
      tokens.push({ type: TOKEN_TYPES.TEXT, value: char, raw: char });
      position += 1;
    }
  }

  return tokens;
}

function serialize(tokens) {
  return tokens.map((t) => t.raw).join("");
}

function findLastToken(tokens, type) {
  for (let i = tokens.length - 1; i >= 0; i--) {
    if (tokens[i].type === type) {
      return { token: tokens[i], index: i };
    }
  }
  return null;
}

function removeToken(tokens, index) {
  return tokens.filter((_, i) => i !== index);
}

function getTokensUpToIndex(tokens, index) {
  return tokens.slice(0, index + 1);
}

export { findLastToken, getTokensUpToIndex, removeToken, serialize, TOKEN_TYPES, tokenize };
