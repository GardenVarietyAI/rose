export const askButton = () => ({
  disabled: false,
  label: "Ask",
  interval: null,
  dots: 0,

  stopSpinner() {
    if (this.interval) clearInterval(this.interval);
    this.interval = null;
    this.dots = 0;
    this.label = "Ask";
  },

  startSpinner() {
    this.stopSpinner();
    this.interval = setInterval(() => {
      this.dots = (this.dots % 3) + 1;
      this.label = "Asking" + ".".repeat(this.dots);
    }, 400);
  },

  getQuery() {
    const form = this.$el.closest("form");
    const queryInput = form?.querySelector('input[name="q"]');
    return queryInput?.value?.trim() || "";
  },

  async ask() {
    const query = this.getQuery();
    if (!query || this.disabled) return;

    this.disabled = true;
    this.startSpinner();

    try {
      const response = await fetch("/v1/chat/completions", {
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
      this.stopSpinner();
      this.disabled = false;
    }
  },
});
