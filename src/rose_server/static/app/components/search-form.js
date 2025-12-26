import { placeCaretEnd } from "../utils/caret.js";
import { isDelimiterKey } from "../utils/keyboard.js";
import { normalizeText, sanitizeText } from "../utils/text.js";
import { findLastToken, removeToken, serialize, TOKEN_TYPES, tokenize } from "../utils/tokenizer.js";

const REGEX_REGEX_ESCAPE = /[.*+?^${}()|[\]\\]/g;

export const searchForm = () => ({
  queryValue: "",
  settingsOpen: false,
  submitting: false,
  mentionOpen: false,
  mentionIndex: 0,
  mentionQuery: "",
  mentionOptions: [],
  lensOptions: [],
  resolveOnNextInput: false,
  idToAtName: {},
  errorMessage: "",

  showError(message) {
    this.errorMessage = message;
    setTimeout(() => this.errorMessage = "", 5000);
  },

  init() {
    const lensMap = this.getLensMap();
    const initialQuery = this.$store.search.query || "";
    const initialLensId = this.$store.search.lens_id || "";
    const initialAtName = Object.entries(lensMap).find(([, id]) => id === initialLensId)?.[0] || "";
    const normalizedQuery = initialAtName
      ? this.stripAtNameFromQuery(initialQuery, initialAtName)
      : initialQuery;

    this.queryValue = normalizedQuery;
    this.buildLensState(lensMap);

    this.$watch("queryValue", (value) => {
      this.$store.search.query = value;
    });
    this.$watch("$store.search.lens_id", (lensId) => {
      const select = this.$refs?.lensSelect;
      if (select) select.value = lensId;
      this.syncEditor({ force: true });
      const editor = this.$refs?.editor;
      if (editor && document.activeElement === editor) {
        placeCaretEnd(editor);
      }
    });
    this.$nextTick(() => {
      this.syncEditor({ force: true });
      if (!initialQuery) {
        this.$refs?.editor?.focus();
      }
    });
  },

  syncEditor({ force = false } = {}) {
    const editor = this.$refs?.editor;
    if (editor && (force || document.activeElement !== editor)) {
      this.renderEditor({ editor });
    }
    const textarea = this.$refs?.textarea;
    if (textarea) textarea.value = this.queryValue;
  },

  syncFromEditor() {
    const editor = this.$refs?.editor;
    if (!editor) {
      throw new Error("Editor ref not found");
    }
    const nextValue = this.serializeEditor(editor);
    if (nextValue !== this.queryValue) {
      this.queryValue = nextValue;
      const textarea = this.$refs?.textarea;
      if (textarea) textarea.value = nextValue;
    }
    const mentionText = this.getMentionText({ editor });
    this.updateMentionState({ text: mentionText });
    if (this.resolveOnNextInput) {
      this.resolveOnNextInput = false;
      this.resolveCompletedMention();
    }
  },

  serializeEditor(editor) {
    const parts = [];
    for (const node of editor.childNodes) {
      if (node.nodeType === Node.TEXT_NODE) {
        const text = normalizeText(node.textContent);
        if (text) parts.push(text);
      } else if (node.nodeType === Node.ELEMENT_NODE && node.dataset?.token !== "lens") {
        const text = normalizeText(node.textContent);
        if (text) parts.push(text);
      }
    }
    return parts.join(" ");
  },

  buildSubmitQuery() {
    const selected = this.getSelectedLens();
    const atName = selected?.atName?.toLowerCase() || "";
    const base = this.queryValue.trim();
    if (!atName) return base;
    return base ? `@${atName} ${base}` : `@${atName}`;
  },

  getLensMap() {
    return this.$store.search.lens_map || {};
  },

  buildLensState(lensMap) {
    const lensOptions = Object.entries(lensMap)
      .map(([atName, lensId]) => ({ lensId, atName: atName.toLowerCase() }))
      .filter((option) => option.atName);
    const idToAtName = Object.fromEntries(
      Object.entries(lensMap).map(([atName, lensId]) => [lensId, atName])
    );
    this.lensOptions = lensOptions;
    this.mentionOptions = lensOptions;
    this.idToAtName = idToAtName;
  },

  getSelectedLens() {
    const lensId = this.$store.search.lens_id || "";
    if (!lensId) return null;
    const atName = this.idToAtName[lensId] || "";
    return { lensId, atName };
  },

  setSelectedLensId(lensId) {
    this.$store.search.lens_id = lensId;
    const select = this.$refs?.lensSelect;
    if (select) select.value = lensId;
  },

  stripAtNameFromQuery(text, atName) {
    if (!text || !atName) return text;
    const escaped = atName.replace(REGEX_REGEX_ESCAPE, "\\$&");
    return normalizeText(text.replace(new RegExp(`@${escaped}`, "gi"), ""));
  },


  renderEditor({ editor } = {}) {
    if (!editor) return;
    const selected = this.getSelectedLens();
    const lensId = selected?.lensId || "";
    const atName = selected?.atName || "";
    const fragment = document.createDocumentFragment();

    if (lensId && atName) {
      const token = document.createElement("span");
      token.setAttribute("x-data", `lensToken('${lensId}', '${atName}')`);
      token.setAttribute("x-on:click", "removeLens()");
      fragment.appendChild(token);
      fragment.appendChild(document.createTextNode(" "));
    }

    fragment.appendChild(document.createTextNode(sanitizeText(this.queryValue)));

    editor.textContent = "";
    editor.appendChild(fragment);
  },

  getMentionText({ editor } = {}) {
    if (!editor) return this.queryValue;
    const selection = window.getSelection();
    let text = editor.textContent || "";
    if (selection && selection.rangeCount) {
      const range = selection.getRangeAt(0).cloneRange();
      range.selectNodeContents(editor);
      range.setEnd(selection.focusNode, selection.focusOffset);
      text = range.toString();
    }
    const selected = this.getSelectedLens();
    const atName = selected?.atName ? selected.atName.toLowerCase() : "";
    if (!atName) return text;
    const prefix = `@${atName} `;
    return text.startsWith(prefix) ? text.slice(prefix.length) : text;
  },

  resetMentionState() {
    this.mentionOpen = false;
    this.mentionQuery = "";
    this.mentionIndex = 0;
    this.mentionOptions = this.lensOptions;
  },

  updateMentionState({ text } = {}) {
    const content = text || "";
    const tokens = tokenize(content);

    if (tokens.length === 0) {
      this.resetMentionState();
      return;
    }

    const lastToken = tokens[tokens.length - 1];

    if (lastToken.type !== TOKEN_TYPES.MENTION) {
      this.resetMentionState();
      return;
    }

    const query = lastToken.value.toLowerCase();
    const options = this.lensOptions.filter((option) =>
      option.atName.startsWith(query)
    );

    this.mentionOpen = options.length > 0;
    this.mentionQuery = query;
    this.mentionOptions = options;
    this.mentionIndex = 0;
  },

  resolveCompletedMention({ lensMap } = {}) {
    const map = lensMap || this.getLensMap();
    const tokens = tokenize(this.queryValue);
    const lastMention = findLastToken(tokens, TOKEN_TYPES.MENTION);

    if (!lastMention) return;

    const lensId = map[lastMention.token.value.toLowerCase()];
    if (!lensId) return;

    this.setSelectedLensId(lensId);
    this.queryValue = serialize(removeToken(tokens, lastMention.index)).trim();
    this.mentionOpen = false;
    this.mentionQuery = "";
    this.syncEditor({ force: true });
    const editor = this.$refs?.editor;
    if (editor) {
      placeCaretEnd(editor);
    }
  },

  selectMention(option) {
    if (!option) return;
    const editor = this.$refs?.editor;
    if (!editor) return;

    const mentionText = this.getMentionText({ editor });
    const normalized = normalizeText(mentionText);
    const tokens = tokenize(normalized);
    const lastMention = findLastToken(tokens, TOKEN_TYPES.MENTION);

    let filteredTokens = tokens;
    if (lastMention && lastMention.token.value.toLowerCase() === this.mentionQuery.toLowerCase()) {
      filteredTokens = removeToken(tokens, lastMention.index);
    }

    const cleanedText = serialize(filteredTokens).trim();

    this.setSelectedLensId(option.lensId);
    this.queryValue = cleanedText ? `${cleanedText} ` : "";
    this.mentionOpen = false;
    this.mentionQuery = "";
    this.syncEditor({ force: true });
    placeCaretEnd(editor);
  },

  handleEditorKeydown(event) {
    if (!this.mentionOpen) {
      if (isDelimiterKey(event.key)) {
        this.resolveOnNextInput = true;
      }
      if (event.key === "Enter") {
        event.preventDefault();
        this.submit();
      }
      return;
    }

    const mentionKeyHandlers = {
      ArrowDown: () => this.mentionIndex = Math.min(this.mentionIndex + 1, this.mentionOptions.length - 1),
      ArrowUp: () => this.mentionIndex = Math.max(this.mentionIndex - 1, 0),
      Enter: () => this.selectMention(this.mentionOptions[this.mentionIndex]),
      " ": () => this.selectMention(this.mentionOptions[this.mentionIndex]),
      Escape: () => this.mentionOpen = false,
    };

    if (mentionKeyHandlers[event.key]) {
      event.preventDefault();
      mentionKeyHandlers[event.key]();
    }
  },

  async submit() {
    if (this.submitting) return;
    const form = this.$refs?.form;
    if (!form) {
      throw new Error("Form ref not found");
    }

    this.submitting = true;

    try {
      const select = this.$refs?.lensSelect;
      const selectedLensId = select?.value?.trim() || this.$store.search.lens_id || "";
      const submitQuery = this.buildSubmitQuery();

      const limit = Number(this.$store.search.limit);
      if (!limit || limit < 1) {
        throw new Error("Invalid search limit");
      }

      const payload = {
        q: submitQuery,
        exact: Boolean(this.$store.search.exact),
        limit,
        lens_id: selectedLensId || undefined,
      };

      const response = await fetch(form.action, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/html",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Search failed (HTTP ${response.status})`);
      }

      const html = await response.text();
      const doc = new DOMParser().parseFromString(html, "text/html");
      const next = doc.querySelector("#search-root");
      const current = document.querySelector("#search-root");

      if (!next) {
        throw new Error("Response missing #search-root element");
      }
      if (!current) {
        throw new Error("Page missing #search-root element");
      }

      current.replaceWith(next);

      const params = new URLSearchParams();
      if (payload.q) params.set("q", payload.q);
      if (payload.exact) params.set("exact", "true");
      if (payload.limit) params.set("limit", String(payload.limit));
      if (payload.lens_id) params.set("lens_id", payload.lens_id);
      const url = params.toString() ? `${form.action}?${params.toString()}` : form.action;
      window.history.replaceState({}, "", url);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.showError(`Search failed: ${message}`);
    } finally {
      this.submitting = false;
    }
  },

  clearLenses() {
    this.$store.search.clearLens();
  },
});
