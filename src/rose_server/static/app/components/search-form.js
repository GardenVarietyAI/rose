import { createBufferController } from "../utils/buffer-controller.js";

export const searchForm = () => ({
  queryValue: "",
  settingsOpen: false,
  submitting: false,
  mentionOpen: false,
  mentionIndex: 0,
  mentionQuery: "",
  mentionOptions: [],
  lensOptions: [],
  idToAtName: {},
  errorMessage: "",
  controller: null,

  showError(message) {
    this.errorMessage = message;
    setTimeout(() => this.errorMessage = "", 5000);
  },

  init() {
    const initialQuery = this.$store.search.query || "";
    const lensMap = this.$store.search.lens_map || {};

    this.queryValue = initialQuery.replace(/^@\S+\s*/, "");
    this.buildLensState(lensMap);
    this.controller = createBufferController(this.$refs?.editor);

    this.$watch("queryValue", (value) => {
      this.$store.search.query = value;
    });
    this.$watch("$store.search.lens_id", (lensId, oldLensId) => {
      const select = this.$refs?.lensSelect;
      if (select) select.value = lensId;

      if (oldLensId !== undefined) {
        this.controller.removeLens(oldLensId);
      }

      this.syncFromStore();

      if (oldLensId !== undefined) {
        if (this.controller && document.activeElement === this.$refs?.editor) {
          this.controller.placeCaretAtEnd();
        }
        if (!lensId) {
          this.submit();
        }
      }
    });
    this.$nextTick(() => {
      this.syncFromStore();
      if (!initialQuery) {
        this.$refs?.editor?.focus();
      }
    });
  },

  syncFromStore() {
    if (!this.controller) return;
    this.controller.loadFromQuery(this.queryValue, this.getSelectedLens());
    const textarea = this.$refs?.textarea;
    if (textarea) textarea.value = this.queryValue;
  },

  syncToStore() {
    if (!this.controller) {
      throw new Error("Buffer controller not initialized");
    }
    this.controller.syncBufferOnly();
    const nextValue = this.controller.serialize();
    if (nextValue !== this.queryValue) {
      this.queryValue = nextValue;
      const textarea = this.$refs?.textarea;
      if (textarea) textarea.value = nextValue;
    }
    this.updateMentionState();
  },

  updateMentionState() {
    if (!this.controller) return;
    const mention = this.controller.detectMention();
    if (!mention) {
      this.mentionOpen = false;
      this.mentionQuery = "";
      this.mentionOptions = this.lensOptions;
      this.mentionIndex = 0;
      return;
    }

    const query = mention.query;
    const options = this.lensOptions.filter((option) =>
      option.atName.startsWith(query)
    );
    const wasOpen = this.mentionOpen && this.mentionQuery === query;
    this.mentionOpen = options.length > 0;
    this.mentionQuery = query;
    this.mentionOptions = options;
    this.mentionIndex = wasOpen ? Math.min(this.mentionIndex, Math.max(options.length - 1, 0)) : 0;
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

  selectMention() {
    if (!this.controller || !this.mentionOpen || !this.mentionOptions.length) return;

    const mention = this.controller.detectMention();
    if (!mention) return;

    const selected = this.mentionOptions[this.mentionIndex];
    if (!selected) return;

    this.controller.insertLensToken(mention, selected.lensId, selected.atName);

    this.$store.search.lens_id = selected.lensId;
    this.queryValue = this.controller.serialize();
    this.mentionOpen = false;
    this.mentionQuery = "";
    this.mentionOptions = this.lensOptions;
    this.mentionIndex = 0;

    this.submit();
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
      this.selectMention();
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
    if (this.submitting) return;
    const form = this.$refs?.form;
    if (!form) {
      throw new Error("Form ref not found");
    }

    this.submitting = true;

    try {
      const select = this.$refs?.lensSelect;
      const selectedLensId = select?.value?.trim() || this.$store.search.lens_id || "";
      const submitQuery = this.queryValue.trim();

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
});
