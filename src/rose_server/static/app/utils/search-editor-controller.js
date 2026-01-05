const MENTION_PATTERN = /(?:^|[^A-Za-z0-9-])([@#])([A-Za-z0-9-]*)/g;

export function buildLensState(lensMap) {
  const lensOptions = Object.entries(lensMap || {})
    .map(([atName, lensId]) => ({ lensId, atName: String(atName || "").toLowerCase() }))
    .filter((option) => option.atName);

  const idToAtName = Object.fromEntries(
    Object.entries(lensMap || {}).map(([atName, lensId]) => [lensId, atName])
  );

  return { lensOptions, idToAtName };
}

export function buildFactsheetState(factsheetMap, factsheetTitleMap) {
  const tagToFactsheetId = {};
  const tagToTitle = {};
  const factsheetIdToTag = {};
  const factsheetIdToTitle = {};
  const options = [];

  for (const [tag, factsheetId] of Object.entries(factsheetMap || {})) {
    if (!tag || !factsheetId) continue;
    const normalizedTag = String(tag).toLowerCase();
    tagToFactsheetId[normalizedTag] = factsheetId;
    const title = factsheetTitleMap?.[tag] || factsheetTitleMap?.[normalizedTag] || "";
    tagToTitle[normalizedTag] = title;
    factsheetIdToTag[factsheetId] = normalizedTag;
    factsheetIdToTitle[factsheetId] = title;
    options.push({ tag: normalizedTag, factsheetId, title: tagToTitle[normalizedTag] });
  }

  return {
    tagOptions: options.sort((a, b) => a.tag.localeCompare(b.tag)),
    tagToFactsheetId,
    tagToTitle,
    factsheetIdToTag,
    factsheetIdToTitle,
  };
}

export function computeMentionState({ controller, lensOptions, tagOptions, previous }) {
  if (!controller) {
    throw new Error("controller is required");
  }

  const mention = controller.detectMention();
  if (!mention) {
    return {
      open: false,
      query: "",
      options: lensOptions,
      index: 0,
      mention: null,
    };
  }

  const query = mention.query;
  const options =
    mention.type === "tag"
      ? tagOptions.filter((option) => option.tag.startsWith(query))
      : lensOptions.filter((option) => option.atName.startsWith(query));

  const wasOpen = Boolean(previous?.open) && previous?.query === query;
  const open = options.length > 0;
  const maxIndex = Math.max(options.length - 1, 0);

  return {
    open,
    query,
    options,
    index: wasOpen ? Math.min(previous?.index ?? 0, maxIndex) : 0,
    mention,
  };
}

function getCaretOffsetWithinEditor(editor) {
  const selection = window.getSelection();
  if (!selection || !selection.rangeCount) return null;

  const range = selection.getRangeAt(0);
  const startContainer = range.startContainer;
  if (!startContainer || !editor.contains(startContainer)) return null;

  const preRange = range.cloneRange();
  preRange.selectNodeContents(editor);
  preRange.setEnd(range.startContainer, range.startOffset);
  return preRange.toString().length;
}

function setCaretOffsetWithinEditor(editor, offset) {
  const selection = window.getSelection();
  if (!selection) return;

  const range = document.createRange();
  const textNode = editor.firstChild;
  if (!textNode || textNode.nodeType !== Node.TEXT_NODE) {
    range.selectNodeContents(editor);
    range.collapse(true);
    selection.removeAllRanges();
    selection.addRange(range);
    return;
  }

  range.setStart(textNode, Math.max(0, Math.min(offset, textNode.textContent?.length || 0)));
  range.collapse(true);
  selection.removeAllRanges();
  selection.addRange(range);
}

export function createSearchEditorController(editor) {
  if (!editor) {
    throw new Error("editor is required");
  }

  return {
    getText() {
      return editor.textContent || "";
    },

    setText(text, caretOffset = null) {
      editor.textContent = text || "";
      if (caretOffset !== null) {
        setCaretOffsetWithinEditor(editor, caretOffset);
      }
    },

    getCaretOffset() {
      return getCaretOffsetWithinEditor(editor);
    },

    detectMention() {
      const text = editor.textContent || "";
      const caretOffset = getCaretOffsetWithinEditor(editor);
      if (caretOffset === null) return null;

      const textBeforeCaret = text.slice(0, caretOffset);
      const matches = Array.from(textBeforeCaret.matchAll(MENTION_PATTERN));
      if (matches.length === 0) return null;

      const lastMatch = matches[matches.length - 1];
      const boundaryLength = lastMatch[0].startsWith("@") || lastMatch[0].startsWith("#") ? 0 : 1;
      const start = lastMatch.index + boundaryLength;
      const end = start + lastMatch[0].length - boundaryLength;

      if (end !== caretOffset) return null;

      return {
        start,
        end,
        query: lastMatch[2].toLowerCase(),
        raw: lastMatch[0].slice(boundaryLength),
        type: lastMatch[1] === "#" ? "tag" : "lens",
      };
    },

    replaceRange(start, end, replacement, caretOffsetAfter = null) {
      const text = editor.textContent || "";
      const before = text.slice(0, start);
      const after = text.slice(end);
      const nextText = `${before}${replacement}${after}`;
      const nextCaret = caretOffsetAfter ?? (before.length + replacement.length);
      editor.textContent = nextText;
      setCaretOffsetWithinEditor(editor, nextCaret);
    },
  };
}
