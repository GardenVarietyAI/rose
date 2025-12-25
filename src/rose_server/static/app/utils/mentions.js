import { tokenizeQuery } from "./query-lexer.js";

export const parseMentions = (text) => {
  const tokens = tokenizeQuery(text);
  return tokens.filter((token) => token.type === "MENTION").map((token) => token.value);
};

export const getLastAtName = (text) => {
  const mentions = parseMentions(text);
  if (!mentions.length) return null;
  return mentions[mentions.length - 1];
};

export const resolveLensId = (text, lensMap) => {
  if (!text || !lensMap) return "";
  const atName = getLastAtName(text);
  return atName && lensMap[atName] ? lensMap[atName] : "";
};
