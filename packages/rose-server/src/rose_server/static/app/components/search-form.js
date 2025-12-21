export const searchForm = () => ({
  value: "",
  isMultiline: false,
  multilineThreshold: 80,

  init() {
    this.value = this.$el?.dataset?.initialQuery || "";
    this.updateMode();
    this.$watch("value", () => this.updateMode());
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

  submit() {
    const form = this.$refs?.form;
    if (!form) return;
    form.requestSubmit();
  },
});
