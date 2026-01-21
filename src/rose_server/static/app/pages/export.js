export function exportPage() {
  return {
    acceptedOnly: true,
    lensId: "",
    generating: false,
    exportId: null,
    stats: null,

    async generateExport() {
      this.generating = true;
      try {
        const response = await fetch("/v1/export/training", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            filters: {
              accepted_only: this.acceptedOnly,
              lens_id: this.lensId || null,
              thread_ids: null,
            },
          }),
        });

        if (!response.ok) {
          throw new Error("Export failed");
        }

        this.stats = await response.json();
        this.exportId = this.stats.export_id;
      } catch (error) {
        console.error("Export error:", error);
        alert("Export failed. Please try again.");
      } finally {
        this.generating = false;
      }
    },

    downloadConversations() {
      window.location.href = `/v1/export/training/${this.exportId}/conversations.jsonl`;
    },
  };
}
