export const askButton = () => ({
  disabled: false,
  label: "Ask",

  getQuery() {
    const form = this.$el.closest("form");
    const queryInput = form?.querySelector('[name="q"]:not([disabled])');
    return queryInput?.value?.trim() || "";
  },

  async ask() {
    const query = this.getQuery();
    if (!query || this.disabled) return;

    this.disabled = true;

    try {
      const response = await fetch("/v1/threads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: [{ role: "user", content: query }] }),
      });

      const data = await response.json();
      if (!data?.thread_id) throw new Error("Missing thread_id");
      window.location.href = "/v1/threads/" + data.thread_id;
    } catch (error) {
      alert("Error: " + (error?.message || String(error)));
    } finally {
      this.disabled = false;
    }
  },
});
