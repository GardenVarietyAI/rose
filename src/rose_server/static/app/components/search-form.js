import { parseQueryModel } from "../utils/query-model.js";
import {
  buildFactsheetState,
  buildLensState,
  computeMentionState,
  createSearchEditorController,
} from "../utils/search-editor-controller.js";
import { debounce } from "../utils/debounce.js";
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
  errorMessage: "",
  editor: null,
  commitContentDebounced: null,

  showError(message) {
    this.errorMessage = message;
    setTimeout(() => this.errorMessage = "", 5000);
  },

  init() {
    const store = this.$store.search;
    const initialQuery = store.content;
    const lensMap = store.lens_map;
    const factsheetMap = store.factsheet_map;
    const factsheetTitleMap = store.factsheet_title_map;

    this.queryValue = initialQuery.replace(/^@\S+\s*/, "");
    const lensState = buildLensState(lensMap);
    this.lensOptions = lensState.lensOptions;
    this.mentionOptions = lensState.lensOptions;
    this.idToAtName = lensState.idToAtName;

    const factsheetState = buildFactsheetState(factsheetMap, factsheetTitleMap);
    this.tagOptions = factsheetState.tagOptions;
    this.factsheetIdToTag = factsheetState.factsheetIdToTag;
    this.factsheetIdToTitle = factsheetState.factsheetIdToTitle;

    this.editor = createSearchEditorController(this.$refs?.editor);
    this.commitContentDebounced = debounce((value) => {
      this.$store.search.setContent(value);
    }, 200);

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
    if (select) select.value = this.$store.search.lens_ids[0] || "";
    this.$store.search.setContent(this.queryValue);
  },

  getQueryModelFromState() {
    const store = this.$store.search;
    return parseQueryModel({
      content: this.queryValue,
      lens_ids: store.lens_ids,
      factsheet_ids: store.factsheet_ids,
      exact: store.exact,
      limit: store.limit,
    });
  },

  async runSearch({ updateUrl } = {}) {
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

      this.$store.search.setContent(submitQuery);
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
    void this.runSearch({ updateUrl: "sq" });
  },

  clearLensSelection() {
    this.$store.search.clearLens();
    const select = this.$refs?.lensSelect;
    if (select) select.value = "";
    this.refreshResults();
  },

  handleLensSelectChange(event) {
    const nextLensId = event?.target?.value?.trim() || "";
    this.$store.search.setLensId(nextLensId);
    this.refreshResults();
  },

  removeFactsheet(factsheetId) {
    this.$store.search.removeFactsheetId(factsheetId);
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
    if (!this.commitContentDebounced) {
      throw new Error("commitContentDebounced not initialized");
    }
    this.commitContentDebounced(this.queryValue);
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
      this.$store.search.addFactsheetId(selected.factsheetId);
    } else {
      this.$store.search.setLensId(selected.lensId);
      const select = this.$refs?.lensSelect;
      if (select) select.value = selected.lensId;
    }

    this.editor.replaceRange(mention.start, mention.end, " ", mention.start + 1);
    this.queryValue = this.editor.getText();
    const textarea = this.$refs?.textarea;
    if (textarea) textarea.value = this.queryValue;

    this.$store.search.setContent(this.queryValue);
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
    await this.runSearch();
  },
});
