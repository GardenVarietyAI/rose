import { parseQueryModel } from "../utils/query-model.js";
import { createSearchEditorController } from "../utils/search-editor-controller.js";
import { computeMentionState } from "../utils/search-mentions.js";
import { buildFactsheetState, buildLensState } from "../utils/search-options.js";
import { submitSearchFragment } from "../utils/search-submit.js";

export const searchForm = () => ({
  queryValue: "",
  settingsOpen: false,
  submitting: false,
  pendingSubmit: false,
  pendingUpdateUrl: false,
  mentionOpen: false,
  mentionIndex: 0,
  mentionQuery: "",
  mentionOptions: [],
  activeMention: null,
  lensOptions: [],
  tagOptions: [],
  idToAtName: {},
  factsheetIdToTag: {},
  factsheetIdToTitle: {},
  selectedLensId: "",
  selectedFactsheetIds: [],
  exact: false,
  limit: 10,
  errorMessage: "",
  editor: null,

  _normalizeFactsheetIds(ids) {
    return Array.from(new Set((ids || []).map((factsheetId) => String(factsheetId).trim()).filter(Boolean)));
  },

  showError(message) {
    this.errorMessage = message;
    setTimeout(() => this.errorMessage = "", 5000);
  },

  init() {
    const initialQuery = this.$store.search.content || "";
    const lensMap = this.$store.search.lens_map || {};
    const factsheetMap = this.$store.search.factsheet_map || {};
    const factsheetTitleMap = this.$store.search.factsheet_title_map || {};

    this.queryValue = initialQuery.replace(/^@\S+\s*/, "");
    const lensState = buildLensState(lensMap);
    this.lensOptions = lensState.lensOptions;
    this.mentionOptions = lensState.lensOptions;
    this.idToAtName = lensState.idToAtName;

    const factsheetState = buildFactsheetState(factsheetMap, factsheetTitleMap);
    this.tagOptions = factsheetState.tagOptions;
    this.factsheetIdToTag = factsheetState.factsheetIdToTag;
    this.factsheetIdToTitle = factsheetState.factsheetIdToTitle;

    const initialLensIds = Array.isArray(this.$store.search.lens_ids) ? this.$store.search.lens_ids : [];
    this.selectedLensId = initialLensIds[0] || "";
    this.selectedFactsheetIds = this._normalizeFactsheetIds(
      Array.isArray(this.$store.search.factsheet_ids) ? this.$store.search.factsheet_ids : []
    );
    this.exact = Boolean(this.$store.search.exact);
    this.limit = Number(this.$store.search.limit) || 10;

    this.editor = createSearchEditorController(this.$refs?.editor);

    this.$nextTick(() => {
      this.seedFromTransport();
      if (!initialQuery) {
        this.$refs?.editor?.focus();
      }
    });
  },

  seedFromTransport() {
    if (!this.editor) return;
    this.editor.setText(this.queryValue);
    const textarea = this.$refs?.textarea;
    if (textarea) textarea.value = this.queryValue;
    const select = this.$refs?.lensSelect;
    if (select) select.value = this.selectedLensId || "";
    this.publishFromState();
  },

  getQueryModelFromState() {
    return parseQueryModel({
      content: this.queryValue,
      lens_ids: this.selectedLensId ? [this.selectedLensId] : [],
      factsheet_ids: this.selectedFactsheetIds,
      exact: this.exact,
      limit: this.limit,
    });
  },

  publishFromState() {
    const store = this.$store?.search;
    if (!store?.applyQueryModel) {
      throw new Error("Search store missing applyQueryModel");
    }
    const model = this.getQueryModelFromState();
    store.applyQueryModel(model);
    const select = this.$refs?.lensSelect;
    if (select) select.value = model.lens_ids[0] || "";
  },

  async runSearch({ updateUrl }) {
    if (this.submitting) {
      this.pendingSubmit = true;
      this.pendingUpdateUrl = this.pendingUpdateUrl || updateUrl;
      return;
    }

    const form = this.$refs?.form;
    if (!form) {
      throw new Error("Form ref not found");
    }

    this.submitting = true;

    try {
      const submitQuery = this.queryValue.trim();
      this.queryValue = submitQuery;
      const textarea = this.$refs?.textarea;
      if (textarea) textarea.value = submitQuery;

      this.publishFromState();
      const model = this.getQueryModelFromState();

      const payload = {
        content: submitQuery,
        exact: model.exact,
        limit: model.limit,
        lens_ids: model.lens_ids,
        factsheet_ids: model.factsheet_ids,
      };

      await submitSearchFragment({
        payload,
        formAction: form.action,
        updateUrl,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.showError(`Search failed: ${message}`);
    } finally {
      this.submitting = false;
      if (this.pendingSubmit) {
        const nextUpdateUrl = this.pendingUpdateUrl;
        this.pendingSubmit = false;
        this.pendingUpdateUrl = false;
        await this.runSearch({ updateUrl: nextUpdateUrl });
      }
    }
  },

  refreshResults() {
    void this.runSearch({ updateUrl: false });
  },

  clearLensSelection() {
    this.selectedLensId = "";
    const select = this.$refs?.lensSelect;
    if (select) select.value = "";
    this.publishFromState();
    this.refreshResults();
  },

  handleLensSelectChange(event) {
    const nextLensId = event?.target?.value?.trim() || "";
    this.selectedLensId = nextLensId;
    this.publishFromState();
    this.refreshResults();
  },

  removeFactsheet(factsheetId) {
    const normalizedId = String(factsheetId).trim();
    this.selectedFactsheetIds = this._normalizeFactsheetIds(
      (this.selectedFactsheetIds || []).filter((candidate) => candidate !== normalizedId)
    );
    this.publishFromState();
    this.refreshResults();
  },

  syncToStore() {
    if (!this.editor) {
      throw new Error("Editor not initialized");
    }
    const nextValue = this.editor.getText();
    if (nextValue !== this.queryValue) {
      this.queryValue = nextValue;
      const textarea = this.$refs?.textarea;
      if (textarea) textarea.value = nextValue;
    }
    this.publishFromState();
    this.updateMentionState();
  },

  updateMentionState() {
    if (!this.editor) return;
    const next = computeMentionState({
      controller: this.editor,
      lensOptions: this.lensOptions,
      tagOptions: this.tagOptions,
      previous: {
        open: this.mentionOpen,
        query: this.mentionQuery,
        index: this.mentionIndex,
      },
    });
    this.mentionOpen = next.open;
    this.mentionQuery = next.query;
    this.mentionOptions = next.options;
    this.mentionIndex = next.index;
    this.activeMention = next.mention;
  },

  selectMention(option = null) {
    if (!this.editor || !this.mentionOpen || !this.mentionOptions.length) return;

    const mention = this.activeMention || this.editor.detectMention();
    if (!mention) return;

    const selected = option || this.mentionOptions[this.mentionIndex];
    if (!selected) return;

    if (mention.type === "tag") {
      const factsheetId = String(selected.factsheetId).trim();
      if (!this.selectedFactsheetIds.includes(factsheetId)) {
        this.selectedFactsheetIds = this._normalizeFactsheetIds([...this.selectedFactsheetIds, factsheetId]);
      }
    } else {
      this.selectedLensId = selected.lensId;
      const select = this.$refs?.lensSelect;
      if (select) select.value = this.selectedLensId;
    }

    this.editor.replaceRange(mention.start, mention.end, " ", mention.start + 1);
    this.queryValue = this.editor.getText();
    const textarea = this.$refs?.textarea;
    if (textarea) textarea.value = this.queryValue;

    this.publishFromState();
    this.mentionOpen = false;
    this.mentionQuery = "";
    this.mentionOptions = this.lensOptions;
    this.mentionIndex = 0;
    this.refreshResults();
  },

  handleEditorKeydown(event) {
    if (!this.mentionOpen) {
      if (event.key === "Enter") {
        event.preventDefault();
        this.submit();
      }
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      const maxIndex = Math.max(this.mentionOptions.length - 1, 0);
      this.mentionIndex = Math.min(this.mentionIndex + 1, maxIndex);
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      this.mentionIndex = Math.max(this.mentionIndex - 1, 0);
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      const selected = this.mentionOptions[this.mentionIndex];
      this.selectMention(selected);
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      this.mentionOpen = false;
      this.mentionQuery = "";
      this.mentionOptions = this.lensOptions;
      this.mentionIndex = 0;
    }
  },

  async submit() {
    await this.runSearch({ updateUrl: true });
  },
});
