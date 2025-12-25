const ENTITY_CHARS = /[\\w-]/;

const isEntityChar = (char) => ENTITY_CHARS.test(char);

const tokenTypeForPrefix = (char) => {
  if (char === "@") return "MENTION";
  if (char === "#") return "TAG";
  if (char === "/") return "COMMAND";
  return null;
};

export const tokenizeQuery = (text) => {
  if (!text) return [];

  const tokens = [];
  let i = 0;
  let textStart = 0;

  while (i < text.length) {
    const char = text[i];
    const tokenType = tokenTypeForPrefix(char);
    const prev = i > 0 ? text[i - 1] : "";
    const boundary = i === 0 || !isEntityChar(prev);

    if (!tokenType || !boundary) {
      i += 1;
      continue;
    }

    let j = i + 1;
    while (j < text.length && isEntityChar(text[j])) {
      j += 1;
    }

    if (j === i + 1) {
      i += 1;
      continue;
    }

    if (textStart < i) {
      tokens.push({
        type: "TEXT",
        value: text.slice(textStart, i),
        start: textStart,
        end: i,
      });
    }

    tokens.push({
      type: tokenType,
      value: text.slice(i + 1, j).toLowerCase(),
      start: i,
      end: j,
    });

    textStart = j;
    i = j;
  }

  if (textStart < text.length) {
    tokens.push({
      type: "TEXT",
      value: text.slice(textStart),
      start: textStart,
      end: text.length,
    });
  }

  return tokens;
};
