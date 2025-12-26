import { placeCaretEnd } from "./caret.js";
import { normalizeText, sanitizeText } from "./text.js";

const REGEX_ZERO_WIDTH_SPACE = /\u200B/g;
const REGEX_NON_BREAKING_SPACE = /\u00A0/g;

export function createEditorController(element) {
  const editor = element;

  return {
    render({ queryText, selectedLens }) {
      if (!editor) return;
      const lensId = selectedLens?.lensId || "";
      const atName = selectedLens?.atName || "";
      const fragment = document.createDocumentFragment();

      if (lensId && atName) {
        const token = document.createElement("span");
        token.setAttribute("x-data", `lensToken('${lensId}', '${atName}')`);
        token.setAttribute("x-on:click", "removeLens()");
        fragment.appendChild(token);
        fragment.appendChild(document.createTextNode(" "));
      }

      fragment.appendChild(document.createTextNode(sanitizeText(queryText || "")));

      editor.textContent = "";
      editor.appendChild(fragment);
    },

    serialize({ normalizeWhitespace = true } = {}) {
      if (!editor) return "";
      const parts = [];
      for (const node of editor.childNodes) {
        if (node.nodeType === Node.ELEMENT_NODE && node.dataset?.token === "lens") {
          continue;
        }
        if (node.nodeType === Node.TEXT_NODE || node.nodeType === Node.ELEMENT_NODE) {
          const rawText = node.textContent || "";
          const text = normalizeWhitespace
            ? normalizeText(rawText)
            : rawText.replace(REGEX_ZERO_WIDTH_SPACE, "").replace(REGEX_NON_BREAKING_SPACE, " ");
          if (text) parts.push(text);
        }
      }
      return normalizeWhitespace ? parts.join(" ") : parts.join("");
    },

    getTextBeforeCaret({ selectedLens } = {}) {
      if (!editor) return "";
      const selection = window.getSelection();
      let text = editor.textContent || "";
      if (selection && selection.rangeCount) {
        const range = selection.getRangeAt(0).cloneRange();
        range.selectNodeContents(editor);
        range.setEnd(selection.focusNode, selection.focusOffset);
        text = range.toString();
      }
      const cleaned = text.replace(REGEX_ZERO_WIDTH_SPACE, "").replace(REGEX_NON_BREAKING_SPACE, " ");
      const atName = selectedLens?.atName ? selectedLens.atName.toLowerCase() : "";
      if (!atName) return cleaned;
      const prefix = `@${atName} `;
      return cleaned.startsWith(prefix) ? cleaned.slice(prefix.length) : cleaned;
    },

    getTextAfterCaret() {
      if (!editor) return "";
      const selection = window.getSelection();
      if (!selection || !selection.rangeCount) return "";
      const range = selection.getRangeAt(0).cloneRange();
      range.selectNodeContents(editor);
      range.setStart(selection.focusNode, selection.focusOffset);
      const text = range.toString();
      return text.replace(REGEX_ZERO_WIDTH_SPACE, "").replace(REGEX_NON_BREAKING_SPACE, " ");
    },

    placeCaretAtEnd() {
      if (!editor) return;
      placeCaretEnd(editor);
    },

    destroy() {},
  };
}
