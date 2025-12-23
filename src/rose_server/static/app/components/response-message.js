export const responseMessage = () => ({
  uuid: null,
  accepted: false,
  collapsed: false,
  expanded: false,
  editing: false,
  draft: "",
  original: "",
  editable: false,
  saving: false,

  init() {
    this.uuid = this.$el.dataset.uuid;
    this.accepted = this.$el.classList.contains("accepted");
    const sourceTemplate = this.$refs?.sourceContent;
    this.editable = Boolean(sourceTemplate);
    if (!this.editable) return;

    const sourceContent = sourceTemplate.content?.textContent;
    this.original = (sourceContent ?? this.$el.querySelector(".message-content")?.innerText ?? "").trimEnd();
    this.draft = this.original;
  },

  startEdit() {
    if (!this.editable) return;
    this.editing = true;
    this.draft = this.original;
    this.$nextTick(() => this.$refs?.editor?.focus());
  },

  cancelEdit() {
    if (!this.editable) return;
    this.editing = false;
    this.draft = this.original;
  },

  async saveEdit() {
    if (!this.editable || this.saving) return;
    this.saving = true;
    try {
      const response = await fetch(`/v1/messages/${this.uuid}/revisions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ content: this.draft }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      window.location.reload();
    } catch (error) {
      console.error("Error:", error);
      this.saving = false;
    }
  },

  async toggleAccepted() {
    try {
      const response = await fetch(`/v1/messages/${this.uuid}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ accepted: !this.accepted }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      this.accepted = !this.accepted;
    } catch (error) {
      console.error("Error:", error);
    }
  },
});
