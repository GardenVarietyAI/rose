import { parseQuery, normalizeTokens, serializeTokens } from "./buffer-parser.js";
import { applyOp, clampCaret } from "./buffer-editor.js";
import { render, caretToDom, domToCaret } from "./buffer-renderer.js";
import { detectMention } from "./mention-detector.js";

export function createBufferController(element) {
  const editor = element;
  let buffer = { tokens: parseQuery(""), caret: { tokenIndex: 0, offset: 0 } };
  let renderCache = null;
  let isRendering = false;

  const syncCaretToDOM = () => {
    if (!editor || !renderCache) return;
    if (document.activeElement !== editor) return;
    const domPos = caretToDom(buffer.caret, renderCache, editor);
    if (!domPos) return;

    const selection = window.getSelection();
    if (!selection) return;

    const range = document.createRange();
    range.setStart(domPos.node, domPos.offset);
    range.collapse(true);
    selection.removeAllRanges();
    selection.addRange(range);
  };

  const syncCaretFromDOM = () => {
    if (!editor || !renderCache) return;

    const selection = window.getSelection();
    if (!selection || !selection.rangeCount) return;

    const range = selection.getRangeAt(0);
    buffer.caret = domToCaret(range.startContainer, range.startOffset, renderCache, editor);
  };

  const syncFromDOM = () => {
    if (!editor) return;

    const rawTokens = [];
    const tempCache = new Map();
    let charOffset = 0;
    let tokenIndex = 0;
    let needsRender = false;

    for (const node of editor.childNodes) {
      if (node.nodeType === Node.ELEMENT_NODE && node.dataset?.token === "lens") {
        rawTokens.push({
          type: "lens",
          lensId: node.dataset.lensId,
          atName: node.dataset.atName
        });
        tempCache.set(tokenIndex++, { node, textStart: charOffset, textEnd: charOffset });
        continue;
      }

      if (node.nodeType === Node.TEXT_NODE) {
        const text = node.textContent || "";
        rawTokens.push({ type: "text", value: text });
        tempCache.set(tokenIndex++, { node, textStart: charOffset, textEnd: charOffset + text.length });
        charOffset += text.length;
        continue;
      }

      if (node.nodeType === Node.ELEMENT_NODE && node.tagName === "BR") {
        needsRender = true;
      }
    }

    const hasAdjacentText = rawTokens.some((token, i) =>
      token.type === "text" && i > 0 && rawTokens[i - 1].type === "text"
    );
    const endsWithLens = rawTokens.length > 0 && rawTokens[rawTokens.length - 1].type === "lens";

    if (hasAdjacentText || endsWithLens || needsRender) {
      needsRender = true;
    }

    let caret = buffer.caret;
    const selection = window.getSelection();
    if (selection && selection.rangeCount) {
      const range = selection.getRangeAt(0);
      caret = domToCaret(range.startContainer, range.startOffset, tempCache, editor);
    }

    if (needsRender) {
      const normalized = normalizeTokens(rawTokens);
      caret = clampCaret(normalized, caret);
      buffer = { tokens: normalized, caret };
      renderBuffer();
    } else {
      const normalized = normalizeTokens(rawTokens);
      caret = clampCaret(normalized, caret);
      buffer = { tokens: normalized, caret };
      renderCache = tempCache;
    }
  };

  const renderBuffer = () => {
    if (!editor || isRendering) return;
    isRendering = true;
    try {
      renderCache = render(buffer.tokens, editor);
      syncCaretToDOM();
      if (window.Alpine) {
        window.Alpine.initTree(editor);
      }
    } finally {
      isRendering = false;
    }
  };

  return {
    getBuffer() {
      return buffer;
    },

    setBuffer(newBuffer) {
      buffer = newBuffer;
      renderBuffer();
    },

    loadFromQuery(queryText, selectedLens) {
      const lensTokens = [];
      if (selectedLens?.lensId && selectedLens?.atName) {
        lensTokens.push({
          lensId: selectedLens.lensId,
          atName: selectedLens.atName
        });
      }
      const tokens = parseQuery(queryText, lensTokens);
      const lastIndex = tokens.length - 1;
      const lastToken = tokens[lastIndex];
      buffer = {
        tokens,
        caret: {
          tokenIndex: lastIndex,
          offset: lastToken?.type === "text" ? lastToken.value.length : 0
        }
      };
      renderBuffer();
    },

    syncFromDOM() {
      syncFromDOM();
    },

    syncBufferOnly() {
      if (!editor) return;

      const rawTokens = [];
      const tempCache = new Map();
      let charOffset = 0;
      let tokenIndex = 0;

      for (const node of editor.childNodes) {
        if (node.nodeType === Node.ELEMENT_NODE && node.dataset?.token === "lens") {
          rawTokens.push({
            type: "lens",
            lensId: node.dataset.lensId,
            atName: node.dataset.atName
          });
          tempCache.set(tokenIndex++, { node, textStart: charOffset, textEnd: charOffset });
        } else if (node.nodeType === Node.TEXT_NODE) {
          const text = node.textContent || "";
          rawTokens.push({ type: "text", value: text });
          tempCache.set(tokenIndex++, { node, textStart: charOffset, textEnd: charOffset + text.length });
          charOffset += text.length;
        }
      }

      if (rawTokens.length === 0) {
        rawTokens.push({ type: "text", value: "" });
        tempCache.set(0, { node: null, textStart: 0, textEnd: 0 });
      }

      const selection = window.getSelection();
      let caret = { tokenIndex: 0, offset: 0 };
      if (selection && selection.rangeCount) {
        const range = selection.getRangeAt(0);
        caret = domToCaret(range.startContainer, range.startOffset, tempCache, editor);
      }

      const maxTokenIndex = rawTokens.length - 1;
      caret.tokenIndex = Math.max(0, Math.min(caret.tokenIndex, maxTokenIndex));
      const token = rawTokens[caret.tokenIndex];
      const maxOffset = token?.type === "text" ? token.value.length : 0;
      caret.offset = Math.max(0, Math.min(caret.offset, maxOffset));

      buffer = { tokens: rawTokens, caret };
      renderCache = tempCache;
    },

    serialize() {
      return serializeTokens(buffer.tokens);
    },

    detectMention() {
      return detectMention(buffer.tokens, buffer.caret);
    },

    insertLensToken(mention, lensId, atName) {
      if (!editor) return;

      const selection = window.getSelection();
      if (!selection || !selection.rangeCount) return;

      const tokenEntry = renderCache.get(mention.tokenIndex);
      if (!tokenEntry || !tokenEntry.node) return;

      const textNode = tokenEntry.node;
      const text = textNode.textContent || "";

      const beforeText = text.slice(0, mention.startOffset);
      const afterText = text.slice(mention.endOffset);

      const lensElement = document.createElement("span");
      lensElement.setAttribute("x-data", `lensToken('${lensId}', '${atName}')`);
      lensElement.setAttribute("x-on:click", "removeLens()");
      lensElement.setAttribute("data-token", "lens");

      const parent = textNode.parentNode;
      if (!parent) return;

      if (beforeText) {
        const beforeNode = document.createTextNode(beforeText);
        parent.insertBefore(beforeNode, textNode);
      }

      parent.insertBefore(lensElement, textNode);

      if (afterText) {
        const afterNode = document.createTextNode(afterText);
        parent.insertBefore(afterNode, textNode);
      } else {
        const spaceNode = document.createTextNode(" ");
        parent.insertBefore(spaceNode, textNode);
      }

      parent.removeChild(textNode);

      if (window.Alpine) {
        window.Alpine.initTree(lensElement);
      }

      const range = document.createRange();
      const nextNode = lensElement.nextSibling;
      if (nextNode && nextNode.nodeType === Node.TEXT_NODE) {
        range.setStart(nextNode, 0);
        range.collapse(true);
        selection.removeAllRanges();
        selection.addRange(range);
      }

      this.syncBufferOnly();
    },

    applyOperation(op) {
      syncCaretFromDOM();
      const result = applyOp(buffer, op);
      buffer = result.buffer;
      renderBuffer();
      return result.inverse;
    },

    removeLens(lensId) {
      if (!editor) return;

      const lensElement = editor.querySelector(`[data-token="lens"][data-lens-id="${lensId}"]`);
      if (!lensElement || !lensElement.parentNode) return;

      lensElement.parentNode.removeChild(lensElement);
      this.syncBufferOnly();
    },

    placeCaretAtEnd() {
      if (!editor) return;

      const selection = window.getSelection();
      if (!selection) return;

      const lastNode = editor.lastChild;
      if (!lastNode) return;

      const range = document.createRange();
      if (lastNode.nodeType === Node.TEXT_NODE) {
        range.setStart(lastNode, lastNode.textContent?.length || 0);
      } else {
        range.setStartAfter(lastNode);
      }
      range.collapse(true);
      selection.removeAllRanges();
      selection.addRange(range);

      this.syncBufferOnly();
    },

    onInput(callback) {
      if (!editor) return;
      editor.addEventListener("input", () => {
        syncCaretFromDOM();
        callback();
      });
    },

    destroy() {
      renderCache = null;
    }
  };
}
