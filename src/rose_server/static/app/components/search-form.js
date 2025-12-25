export const searchForm = () => ({
  value: "",
  isMultiline: false,
  multilineThreshold: 80,
  settingsOpen: false,
  submitting: false,

  init() {
    const initialQuery = this.$el?.dataset?.initialQuery || "";
    const initialLensId = this.$el?.dataset?.initialLensId || "";
    this.$store.search.query = initialQuery;
    this.$store.search.lensId = initialLensId;
    this.value = initialQuery;
    this.updateMode();
    this.$watch("value", () => this.updateMode());
    this.$watch("value", (value) => {
      this.$store.search.query = value;
    });
  },

  updateMode() {
    const shouldBeMultiline = this.value.length > this.multilineThreshold || this.value.includes("\n");
    if (shouldBeMultiline === this.isMultiline) return;
    this.isMultiline = shouldBeMultiline;
    this.$nextTick(() => {
      const el = this.isMultiline ? this.$refs?.multi : this.$refs?.single;
      if (!el) return;
      el.focus();
      if (typeof el.setSelectionRange === "function") el.setSelectionRange(el.value.length, el.value.length);
      if (this.isMultiline) this.autogrow(el);
    });
  },

  handlePaste(event) {
    const text = event?.clipboardData?.getData("text") || "";
    if (!text.includes("\n")) return;
    event.preventDefault();

    const target = event.target;
    const start = typeof target?.selectionStart === "number" ? target.selectionStart : this.value.length;
    const end = typeof target?.selectionEnd === "number" ? target.selectionEnd : this.value.length;
    this.value = `${this.value.slice(0, start)}${text}${this.value.slice(end)}`;

    this.isMultiline = true;
    this.$nextTick(() => {
      const textarea = this.$refs?.multi;
      if (!textarea) return;
      textarea.focus();
      if (typeof textarea.setSelectionRange === "function") {
        const pos = start + text.length;
        textarea.setSelectionRange(pos, pos);
      }
      this.autogrow(textarea);
    });
  },

  autogrow(target) {
    if (!target) return;
    target.style.height = "auto";
    target.style.height = `${target.scrollHeight}px`;
  },

  async submit() {
    if (this.submitting) return;
    const form = this.$refs?.form;
    if (!form || typeof fetch !== "function") {
      form?.requestSubmit();
      return;
    }

    this.submitting = true;

    try {
      const payload = {
        q: this.$store.search.query || "",
        exact: Boolean(this.$store.search.exact),
        limit: Number(this.$store.search.limit) || 10,
        lens_id: this.$store.search.lensId || undefined,
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
        throw new Error("Search failed");
      }

      const html = await response.text();
      const doc = new DOMParser().parseFromString(html, "text/html");
      const next = doc.querySelector("#search-root");
      const current = document.querySelector("#search-root");

      if (next && current) {
        if (window.Alpine) window.Alpine.destroyTree(current);
        current.replaceWith(next);
        if (window.Alpine) window.Alpine.initTree(next);
      } else {
        window.location.href = form.action;
      }

      const params = new URLSearchParams();
      if (payload.q) params.set("q", payload.q);
      if (payload.exact) params.set("exact", "true");
      if (payload.limit && payload.limit !== 10) params.set("limit", String(payload.limit));
      if (payload.lens_id) params.set("lens_id", payload.lens_id);
      const url = params.toString() ? `${form.action}?${params.toString()}` : form.action;
      window.history.replaceState({}, "", url);
    } catch (error) {
      form.requestSubmit();
    } finally {
      this.submitting = false;
    }
  },

  clearLenses() {
    this.$store.search.lensId = "";
    const select = this.$refs?.lensSelect;
    if (!select) return;
    select.value = "";
  },
});
