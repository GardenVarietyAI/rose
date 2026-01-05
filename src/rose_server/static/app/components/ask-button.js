export const askButton = () => ({
  disabled: false,
  label: "Ask",

  async ask() {
    const query = this.$el.closest("form")?.querySelector('[name="q"]:not([disabled])')?.value?.trim() || "";
    if (!query || this.disabled) return;

    const lensIds = this.$store?.search?.lens_ids;
    const lensId = Array.isArray(lensIds) ? lensIds[0] || "" : "";
    const factsheetIds = this.$store?.search?.factsheet_ids;
    const normalizedFactsheetIds = Array.isArray(factsheetIds) ? factsheetIds : [];

    this.disabled = true;

    try {
      const response = await fetch("/v1/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: query,
          lens_ids: lensId ? [lensId] : [],
          factsheet_ids: normalizedFactsheetIds,
        }),
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
