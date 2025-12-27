import { highlightCode } from "./highlight.js";

function escapeHtml(s) {
  if (typeof s !== "string") {
    throw new TypeError("escapeHtml expects a string");
  }
  return s.replace(/[&<>"']/g, (c) => (
    c === "&" ? "&amp;" :
    c === "<" ? "&lt;" :
    c === ">" ? "&gt;" :
    c === '"' ? "&quot;" : "&#39;"
  ));
}

function parseInline(s) {
  if (typeof s !== "string") {
    throw new TypeError("parseInline expects a string");
  }

  const codeSpans = [];
  const links = [];
  const autoLinks = [];

  const codeSpanToken = (i) => `\u0001ROSECODE${i}\u0001`;
  const linkToken = (i) => `\u0001ROSELINK${i}\u0001`;
  const autoLinkToken = (i) => `\u0001ROSEAUTOLINK${i}\u0001`;

  // inline code first prevents formatting inside code
  s = s.replace(/`([^`]*?)`/g, (_, code) => {
    const i = codeSpans.length;
    codeSpans.push(escapeHtml(code));
    return codeSpanToken(i);
  });

  // links: [text](url) with scheme validation
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, text, url) => {
    const trimmedUrl = url.trim();

    // validate URL scheme (prevent javascript:, data:, etc.)
    const isValidScheme = /^(https?|mailto|tel):/i.test(trimmedUrl) || /^[/.#]/.test(trimmedUrl);
    if (!isValidScheme) {
      return text;
    }

    const i = links.length;
    const safeText = escapeHtml(text);
    const safeUrl = escapeHtml(trimmedUrl);
    links.push(`<a href="${safeUrl}" target="_blank" rel="noopener noreferrer">${safeText}</a>`);
    return linkToken(i);
  });

  // auto-link bare URLs (common in model output)
  s = s.replace(/\bhttps?:\/\/[^\s<>"']+/g, (rawUrl) => {
    let url = rawUrl;
    let trailing = "";
    while (/[),.;:!?\\\]]$/.test(url)) {
      trailing = url.slice(-1) + trailing;
      url = url.slice(0, -1);
    }
    if (!url) {
      return escapeHtml(rawUrl);
    }

    const i = autoLinks.length;
    const safeUrl = escapeHtml(url);
    autoLinks.push(`<a href="${safeUrl}" target="_blank" rel="noopener noreferrer">${safeUrl}</a>${escapeHtml(trailing)}`);
    return autoLinkToken(i);
  });

  // escape before inserting markup to prevent injection
  s = escapeHtml(s);

  // bold/italic.
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  s = s.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  s = s.replace(/_([^_]+)_/g, "<em>$1</em>");

  // restore link placeholders
  s = s.replace(/\u0001ROSELINK(\d+)\u0001/g, (_, i) => {
    const linkHtml = links[Number(i)];
    if (linkHtml === undefined) {
      throw new Error("Invalid link placeholder index");
    }
    return linkHtml;
  });

  // restore auto-link placeholders
  s = s.replace(/\u0001ROSEAUTOLINK(\d+)\u0001/g, (_, i) => {
    const linkHtml = autoLinks[Number(i)];
    if (linkHtml === undefined) {
      throw new Error("Invalid auto-link placeholder index");
    }
    return linkHtml;
  });

  // restore code span placeholders
  s = s.replace(/\u0001ROSECODE(\d+)\u0001/g, (_, i) => {
    const codeHtml = codeSpans[Number(i)];
    if (codeHtml === undefined) {
      throw new Error("Invalid code placeholder index");
    }
    return `<code>${codeHtml}</code>`;
  });

  return s;
}

function markdownToHtml(md) {
  if (typeof md !== "string") {
    throw new TypeError("markdownToHtml expects a string");
  }

  const MAX_INPUT_LENGTH = 100000;
  if (md.length > MAX_INPUT_LENGTH) {
    const truncated = md.slice(0, MAX_INPUT_LENGTH);
    let warning = `<div class="markdown-error">Content exceeds ${MAX_INPUT_LENGTH} characters, showing truncated...`;
    warning += `</div><pre>${escapeHtml(truncated)}</pre>`;
    return warning;
  }

  md = md.replace(/\r\n?/g, "\n");

  const lines = md.split("\n");
  let html = "";
  let inUl = false, inOl = false, inBlockquote = false, inCodeFence = false;
  let paraBuffer = [];
  let quoteBuffer = [];
  let codeFenceLang = "";
  let codeFenceBuffer = [];

  function flushParagraph() {
    if (paraBuffer.length > 0) {
      html += `<p>${parseInline(paraBuffer.join(" "))}</p>\n`;
      paraBuffer = [];
    }
  }

  function flushBlockquote() {
    if (!inBlockquote) {
      return;
    }
    html += `<blockquote>${quoteBuffer.join("<br />\n")}</blockquote>\n`;
    quoteBuffer = [];
    inBlockquote = false;
  }

  function flushCodeFence() {
    if (!inCodeFence) {
      return;
    }
    const code = codeFenceBuffer.join("\n");
    const cls = codeFenceLang ? ` class="language-${escapeHtml(codeFenceLang)}"` : "";
    const highlighted = highlightCode(code, codeFenceLang);
    html += `<pre><code${cls}>${highlighted}</code></pre>\n`;
    codeFenceLang = "";
    codeFenceBuffer = [];
    inCodeFence = false;
  }

  function closeLists() {
    if (inUl) { html += "</ul>\n"; inUl = false; }
    if (inOl) { html += "</ol>\n"; inOl = false; }
  }

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];

    // fenced code block
    const mFence = line.match(/^```([^\s`]+)?\s*$/);
    if (mFence) {
      flushParagraph();
      closeLists();
      flushBlockquote();
      if (inCodeFence) {
        flushCodeFence();
      } else {
        inCodeFence = true;
        codeFenceLang = (mFence[1] || "").trim();
        codeFenceBuffer = [];
      }
      continue;
    }
    if (inCodeFence) {
      codeFenceBuffer.push(line);
      continue;
    }

    // blank line
    if (!line.trim()) {
      flushParagraph();
      closeLists();
      flushBlockquote();
      continue;
    }

    // hr
    if (/^(\-\-\-|\*\*\*|___)\s*$/.test(line.trim())) {
      flushParagraph();
      closeLists();
      flushBlockquote();
      html += "<hr />\n";
      continue;
    }

    // blockquote
    if (/^\s*>\s?/.test(line)) {
      flushParagraph();
      closeLists();
      if (!inBlockquote) {
        inBlockquote = true;
      }
      const content = line.replace(/^\s*>\s?/, "");
      quoteBuffer.push(parseInline(content));
      continue;
    }

    // headings
    const mH = line.match(/^(#{1,6})\s+(.*)$/);
    if (mH) {
      flushParagraph();
      closeLists();
      flushBlockquote();
      const level = mH[1].length;
      const content = mH[2].trim();
      html += `<h${level}>${parseInline(content)}</h${level}>\n`;
      continue;
    }

    // ordered list
    const mOl = line.match(/^\s*(\d+)\.\s+(.*)$/);
    if (mOl) {
      flushParagraph();
      flushBlockquote();
      if (inUl) { html += "</ul>\n"; inUl = false; }
      if (!inOl) { html += "<ol>\n"; inOl = true; }
      html += `<li>${parseInline(mOl[2])}</li>\n`;
      continue;
    }

    // unordered list
    const mUl = line.match(/^\s*[-*+]\s+(.*)$/);
    if (mUl) {
      flushParagraph();
      flushBlockquote();
      if (inOl) { html += "</ol>\n"; inOl = false; }
      if (!inUl) { html += "<ul>\n"; inUl = true; }
      html += `<li>${parseInline(mUl[1])}</li>\n`;
      continue;
    }

    // paragraph continuation
    flushBlockquote();
    closeLists();
    paraBuffer.push(line);
  }

  // if a fence was opened but never closed, render the rest as code
  flushCodeFence();
  flushParagraph();
  closeLists();
  flushBlockquote();
  return html.trim();
}

export { markdownToHtml };
