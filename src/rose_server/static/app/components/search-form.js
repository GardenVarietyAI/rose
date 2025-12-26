import { isDelimiterKey } from "../utils/keyboard.js";
import { normalizeText } from "../utils/text.js";
import { createEditorController } from "../utils/editor-controller.js";
import { createMentionEngine } from "../utils/mention-engine.js";
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
  editor: null,
  mentions: null,

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
    this.editor = createEditorController(this.$refs?.editor);
    this.mentions = createMentionEngine();

    this.$watch("queryValue", (value) => {
      this.$store.search.query = value;
    });
    this.$watch("$store.search.lens_id", (lensId) => {
      const select = this.$refs?.lensSelect;
      if (select) select.value = lensId;
      this.syncEditor({ force: true });
      if (this.editor && document.activeElement === this.$refs?.editor) {
        this.editor.placeCaretAtEnd();
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
    if (editor && this.editor && (force || document.activeElement !== editor)) {
      this.editor.render({ queryText: this.queryValue, selectedLens: this.getSelectedLens() });
    }
    const textarea = this.$refs?.textarea;
    if (textarea) textarea.value = this.queryValue;
  },

  syncFromEditor() {
    if (!this.editor) {
      throw new Error("Editor controller not initialized");
    }
    const nextValue = this.editor.serialize({ normalizeWhitespace: true });
    if (nextValue !== this.queryValue) {
      this.queryValue = nextValue;
      const textarea = this.$refs?.textarea;
      if (textarea) textarea.value = nextValue;
    }
    const mentionText = this.editor.getTextBeforeCaret({ selectedLens: this.getSelectedLens() });
    const mentionState = this.mentions.update(mentionText, this.lensOptions);
    this.mentionOpen = mentionState.open;
    this.mentionQuery = mentionState.query;
    this.mentionOptions = mentionState.options;
    this.mentionIndex = mentionState.index;
    if (this.resolveOnNextInput) {
      this.resolveOnNextInput = false;
      this.resolveCompletedMention();
    }
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

  resolveCompletedMention({ lensMap } = {}) {
    const map = lensMap || this.getLensMap();
    if (!this.editor) return;
    const selected = this.getSelectedLens();
    const textBeforeCaret = this.editor.getTextBeforeCaret({ selectedLens: selected });
    const beforeTokens = tokenize(textBeforeCaret);
    const lastMention = findLastToken(beforeTokens, TOKEN_TYPES.MENTION);
    if (!lastMention) return;

    const lensId = map[lastMention.token.value.toLowerCase()];
    if (!lensId) return;

    const cleanedBefore = serialize(removeToken(beforeTokens, lastMention.index)).trim();

    this.setSelectedLensId(lensId);
    this.queryValue = cleanedBefore;
    this.mentions?.reset();
    this.syncEditor({ force: true });
    this.editor?.placeCaretAtEnd();
  },

  selectMention() {
    if (!this.mentions || !this.editor) return;

    const selection = this.mentions.select();
    if (!selection) return;

    const selected = this.getSelectedLens();
    const textBeforeCaret = this.editor.getTextBeforeCaret({ selectedLens: selected });
    const tokens = tokenize(textBeforeCaret);
    const lastMention = findLastToken(tokens, TOKEN_TYPES.MENTION);

    let cleanedBefore = textBeforeCaret;
    if (lastMention && lastMention.token.raw === selection.matchedToken) {
      const filteredTokens = removeToken(tokens, lastMention.index);
      cleanedBefore = serialize(filteredTokens).trim();
    }

    this.setSelectedLensId(selection.option.lensId);
    this.queryValue = cleanedBefore ? `${cleanedBefore} ` : "";
    this.mentions.reset();
    this.syncEditor({ force: true });
    this.editor.placeCaretAtEnd();
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

    if (!this.mentions) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      this.mentionIndex = this.mentions.navigate("down");
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      this.mentionIndex = this.mentions.navigate("up");
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      this.selectMention();
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      this.mentionOpen = false;
      this.mentionQuery = "";
      this.mentionOptions = this.lensOptions;
      this.mentionIndex = 0;
      this.mentions?.reset();
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
