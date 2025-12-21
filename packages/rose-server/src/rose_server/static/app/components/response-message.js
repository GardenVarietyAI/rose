export const responseMessage = () => ({
  uuid: null,
  accepted: false,

  init() {
    this.uuid = this.$el.dataset.uuid;
    this.accepted = this.$el.classList.contains("accepted");
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
