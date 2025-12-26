import { normalizeTokens } from "./buffer-parser.js";

export function clampCaret(tokens, caret) {
  const tokenIndex = Math.max(0, Math.min(caret.tokenIndex, tokens.length - 1));
  const token = tokens[tokenIndex];
  const maxOffset = token.type === "text" ? token.value.length : 0;
  const offset = Math.max(0, Math.min(caret.offset, maxOffset));
  return { tokenIndex, offset };
}

export function applyOp(buffer, op) {
  const { tokens, caret } = buffer;

  if (op.type === "insertText") {
    const { at, text } = op;
    const newTokens = [...tokens];
    const targetToken = newTokens[at.tokenIndex];

    if (targetToken.type !== "text") {
      throw new Error("Cannot insert text into non-text token");
    }

    const before = targetToken.value.slice(0, at.offset);
    const after = targetToken.value.slice(at.offset);
    newTokens[at.tokenIndex] = { type: "text", value: before + text + after };

    const normalized = normalizeTokens(newTokens);
    const newCaret = { tokenIndex: at.tokenIndex, offset: at.offset + text.length };
    const clamped = clampCaret(normalized, newCaret);

    const inverse = {
      type: "deleteRange",
      from: at,
      to: { tokenIndex: at.tokenIndex, offset: at.offset + text.length }
    };

    return {
      buffer: { tokens: normalized, caret: clamped },
      inverse
    };
  }

  if (op.type === "deleteRange") {
    const { from, to } = op;
    const newTokens = [...tokens];
    const deletedTokens = [];

    if (from.tokenIndex === to.tokenIndex) {
      const targetToken = newTokens[from.tokenIndex];

      if (targetToken.type !== "text") {
        throw new Error("Cannot delete from non-text token");
      }

      const before = targetToken.value.slice(0, from.offset);
      const deleted = targetToken.value.slice(from.offset, to.offset);
      const after = targetToken.value.slice(to.offset);
      newTokens[from.tokenIndex] = { type: "text", value: before + after };

      const normalized = normalizeTokens(newTokens);
      const newCaret = { tokenIndex: from.tokenIndex, offset: from.offset };
      const clamped = clampCaret(normalized, newCaret);

      const inverse = {
        type: "insertText",
        at: from,
        text: deleted
      };

      return {
        buffer: { tokens: normalized, caret: clamped },
        inverse
      };
    }

    const fromToken = newTokens[from.tokenIndex];
    const toToken = newTokens[to.tokenIndex];

    if (fromToken.type !== "text" || toToken.type !== "text") {
      throw new Error("Cross-token delete requires text tokens at boundaries");
    }

    const beforeStart = fromToken.value.slice(0, from.offset);
    const deletedStart = fromToken.value.slice(from.offset);
    const deletedEnd = toToken.value.slice(0, to.offset);
    const afterEnd = toToken.value.slice(to.offset);

    for (let i = from.tokenIndex; i <= to.tokenIndex; i++) {
      deletedTokens.push(newTokens[i]);
    }

    deletedTokens[0] = { type: "text", value: deletedStart };
    deletedTokens[deletedTokens.length - 1] = { type: "text", value: deletedEnd };

    const mergedDeleted = [];
    for (let i = 0; i < deletedTokens.length; i++) {
      const token = deletedTokens[i];
      if (token.type === "text") {
        if (mergedDeleted.length > 0 && mergedDeleted[mergedDeleted.length - 1].type === "text") {
          const prev = mergedDeleted[mergedDeleted.length - 1];
          mergedDeleted[mergedDeleted.length - 1] = {
            type: "text",
            value: prev.value + token.value
          };
        } else {
          mergedDeleted.push(token);
        }
      } else {
        mergedDeleted.push(token);
      }
    }

    for (let i = to.tokenIndex; i >= from.tokenIndex; i--) {
      newTokens.splice(i, 1);
    }

    newTokens.splice(from.tokenIndex, 0, { type: "text", value: beforeStart + afterEnd });

    const normalized = normalizeTokens(newTokens);
    const newCaret = { tokenIndex: from.tokenIndex, offset: from.offset };
    const clamped = clampCaret(normalized, newCaret);

    const inverse = {
      type: "insertTokens",
      at: from,
      tokens: mergedDeleted
    };

    return {
      buffer: { tokens: normalized, caret: clamped },
      inverse
    };
  }

  if (op.type === "replaceToken") {
    const { index, token } = op;
    const newTokens = [...tokens];
    const oldToken = newTokens[index];
    newTokens[index] = token;

    const normalized = normalizeTokens(newTokens);
    const clamped = clampCaret(normalized, caret);

    const inverse = {
      type: "replaceToken",
      index,
      token: oldToken
    };

    return {
      buffer: { tokens: normalized, caret: clamped },
      inverse
    };
  }

  if (op.type === "insertTokens") {
    const { at, tokens: tokensToInsert } = op;
    const newTokens = [...tokens];
    const targetToken = newTokens[at.tokenIndex];

    if (targetToken.type !== "text") {
      throw new Error("Cannot insert tokens into non-text token");
    }

    const before = targetToken.value.slice(0, at.offset);
    const after = targetToken.value.slice(at.offset);

    const insertedTokens = [...tokensToInsert];

    if (insertedTokens.length > 0) {
      const firstToken = insertedTokens[0];
      if (firstToken.type === "text") {
        insertedTokens[0] = { type: "text", value: before + firstToken.value };
      } else if (before) {
        insertedTokens.unshift({ type: "text", value: before });
      }

      const lastToken = insertedTokens[insertedTokens.length - 1];
      if (lastToken.type === "text") {
        insertedTokens[insertedTokens.length - 1] = { type: "text", value: lastToken.value + after };
      } else if (after) {
        insertedTokens.push({ type: "text", value: after });
      }
    } else {
      insertedTokens.push({ type: "text", value: before + after });
    }

    newTokens.splice(at.tokenIndex, 1, ...insertedTokens);

    const normalized = normalizeTokens(newTokens);

    const lastInserted = tokensToInsert[tokensToInsert.length - 1];
    let endTokenIndex = 0;
    let endOffset = 0;

    if (lastInserted && lastInserted.type !== "text") {
      let insertionPointChars = 0;
      for (let i = 0; i < at.tokenIndex; i++) {
        const token = tokens[i];
        if (token.type === "text") {
          insertionPointChars += token.value.length;
        }
      }
      insertionPointChars += at.offset;

      let targetChars = insertionPointChars;
      for (const token of tokensToInsert) {
        if (token.type === "text") {
          targetChars += token.value.length;
        }
      }

      let lensCountToInsert = 0;
      for (const token of tokensToInsert) {
        if (token.type !== "text") lensCountToInsert++;
      }

      let charsSeen = 0;
      let lensSeenAfterInsertion = 0;
      let passedInsertionPoint = false;

      for (let i = 0; i < normalized.length; i++) {
        const token = normalized[i];
        if (token.type === "text") {
          if (!passedInsertionPoint && charsSeen >= insertionPointChars) {
            passedInsertionPoint = true;
          }
          if (targetChars <= charsSeen + token.value.length) {
            endTokenIndex = i;
            endOffset = targetChars - charsSeen;
            break;
          }
          charsSeen += token.value.length;
        } else {
          if (passedInsertionPoint) {
            lensSeenAfterInsertion++;
            if (lensSeenAfterInsertion === lensCountToInsert) {
              endTokenIndex = i + 1;
              endOffset = 0;
              break;
            }
          }
        }
      }
    } else {
      let targetChars = before.length;
      for (const token of tokensToInsert) {
        if (token.type === "text") {
          targetChars += token.value.length;
        }
      }

      let charsSeen = 0;
      for (let i = 0; i < normalized.length; i++) {
        const token = normalized[i];
        if (token.type === "text") {
          if (targetChars <= charsSeen + token.value.length) {
            endTokenIndex = i;
            endOffset = targetChars - charsSeen;
            break;
          }
          charsSeen += token.value.length;
        }
      }
    }

    const newCaret = { tokenIndex: endTokenIndex, offset: endOffset };
    const clamped = clampCaret(normalized, newCaret);

    const inverse = {
      type: "deleteRange",
      from: at,
      to: clamped
    };

    return {
      buffer: { tokens: normalized, caret: clamped },
      inverse
    };
  }

  throw new Error(`Unknown op type: ${op.type}`);
}
