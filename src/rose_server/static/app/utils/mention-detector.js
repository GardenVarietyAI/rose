const MENTION_PATTERN = /(?:^|[^A-Za-z0-9-])@([A-Za-z0-9-]*)/g;

export function detectMention(tokens, caret) {
  let mergedText = "";
  let tokenOffsets = [];
  let caretCharIndex = 0;
  let caretTokenIndex = -1;

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (token.type === "text") {
      const startChar = mergedText.length;
      mergedText += token.value;
      tokenOffsets.push({ tokenIndex: i, startChar, endChar: mergedText.length });

      if (i < caret.tokenIndex) {
        caretCharIndex += token.value.length;
      } else if (i === caret.tokenIndex) {
        caretCharIndex += caret.offset;
        caretTokenIndex = tokenOffsets.length - 1;
      }
    } else {
      tokenOffsets.push({ tokenIndex: i, startChar: mergedText.length, endChar: mergedText.length });
      if (i < caret.tokenIndex) {
        caretTokenIndex = tokenOffsets.length - 1;
      }
    }
  }

  if (caretTokenIndex === -1 || mergedText.length === 0) {
    return null;
  }

  const textBeforeCaret = mergedText.slice(0, caretCharIndex);
  const matches = Array.from(textBeforeCaret.matchAll(MENTION_PATTERN));
  if (matches.length === 0) return null;

  const lastMatch = matches[matches.length - 1];
  const boundaryLength = lastMatch[0].startsWith("@") ? 0 : 1;
  const mentionStartChar = lastMatch.index + boundaryLength;
  const mentionEndChar = mentionStartChar + lastMatch[0].length - boundaryLength;

  if (mentionEndChar !== caretCharIndex) {
    return null;
  }

  for (const offset of tokenOffsets) {
    if (mentionStartChar >= offset.startChar && mentionStartChar < offset.endChar) {
      if (mentionEndChar > offset.endChar) {
        return null;
      }
      return {
        tokenIndex: offset.tokenIndex,
        startOffset: mentionStartChar - offset.startChar,
        endOffset: mentionEndChar - offset.startChar,
        query: lastMatch[1].toLowerCase(),
        raw: lastMatch[0].slice(boundaryLength)
      };
    }
  }

  return null;
}
