export const askButton = () => ({
  disabled: false,
  label: "Ask",

  async ask() {
    const query = this.$el.closest("form")?.querySelector('[name="q"]:not([disabled])')?.value?.trim() || "";
    if (!query || this.disabled) return;

    this.disabled = true;

    try {
      const form = this.$el.closest("form");
      const lensIdValue = form?.querySelector('select[name="lens_id"]')?.value?.trim();
      let lensId = lensIdValue || "";
      const response = await fetch("/v1/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: query,
          lens_id: lensId || undefined,
        }),
      });

      const data = await response.json();
      if (!data?.thread_id) throw new Error("Missing thread_id");
      const lensSuffix = lensId ? `?lens_id=${encodeURIComponent(lensId)}` : "";
      window.location.href = "/v1/threads/" + data.thread_id + lensSuffix;
    } catch (error) {
      alert("Error: " + (error?.message || String(error)));
    } finally {
      this.disabled = false;
    }
  },
});
