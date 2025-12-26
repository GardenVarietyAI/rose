import { sanitizeText } from "./text.js";

export function render(tokens, container) {
  container.textContent = "";
  const cache = new Map();
  let charOffset = 0;

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];

    if (token.type === "lens") {
      const span = document.createElement("span");
      span.setAttribute("x-data", `lensToken('${token.lensId}', '${token.atName}')`);
      span.setAttribute("x-on:click", "removeLens()");
      span.setAttribute("data-token", "lens");
      container.appendChild(span);

      cache.set(i, {
        node: span,
        textStart: charOffset,
        textEnd: charOffset
      });
    } else if (token.type === "text") {
      const textNode = document.createTextNode(sanitizeText(token.value));
      container.appendChild(textNode);

      cache.set(i, {
        node: textNode,
        textStart: charOffset,
        textEnd: charOffset + token.value.length
      });

      charOffset += token.value.length;
    }
  }

  return cache;
}

export function caretToDom(caret, cache, container) {
  const entry = cache.get(caret.tokenIndex);
  if (!entry) return null;

  if (entry.node.nodeType === Node.TEXT_NODE) {
    return { node: entry.node, offset: caret.offset };
  }

  if (caret.offset === 0) {
    return { node: container, offset: Array.from(container.childNodes).indexOf(entry.node) };
  }

  const nextIndex = caret.tokenIndex + 1;
  const nextEntry = cache.get(nextIndex);
  if (nextEntry) {
    return { node: container, offset: Array.from(container.childNodes).indexOf(nextEntry.node) };
  }

  return { node: container, offset: container.childNodes.length };
}

export function domToCaret(node, offset, cache, container) {
  for (const [tokenIndex, entry] of cache.entries()) {
    if (entry.node === node) {
      if (node.nodeType === Node.TEXT_NODE) {
        return { tokenIndex, offset: Math.min(offset, entry.textEnd - entry.textStart) };
      }
      return { tokenIndex, offset: 0 };
    }
  }

  if (node === container) {
    const childNodes = Array.from(container.childNodes);
    if (offset >= childNodes.length) {
      const lastIndex = cache.size - 1;
      const lastEntry = cache.get(lastIndex);
      if (lastEntry && lastEntry.node.nodeType === Node.TEXT_NODE) {
        return { tokenIndex: lastIndex, offset: lastEntry.textEnd - lastEntry.textStart };
      }
      return { tokenIndex: lastIndex, offset: 0 };
    }

    const targetNode = childNodes[offset];
    for (const [tokenIndex, entry] of cache.entries()) {
      if (entry.node === targetNode) {
        return { tokenIndex, offset: 0 };
      }
    }
  }

  return { tokenIndex: cache.size - 1, offset: 0 };
}
