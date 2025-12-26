const REGEX_ZERO_WIDTH_SPACE = /\u200B/g;
const REGEX_NON_BREAKING_SPACE = /\u00A0/g;
const REGEX_WHITESPACE = /\s+/g;

export function normalizeText(text) {
  return (text || "")
    .replace(REGEX_ZERO_WIDTH_SPACE, "")
    .replace(REGEX_NON_BREAKING_SPACE, " ")
    .replace(REGEX_WHITESPACE, " ")
    .trim();
}

export function sanitizeText(text) {
  const div = document.createElement("div");
  div.textContent = text || "";
  return div.textContent || "";
}
