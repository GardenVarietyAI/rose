export function exportPage() {
  return {
    acceptedOnly: true,
    lensId: "",
    splitRatio: 0.9,
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
            split_ratio: this.splitRatio,
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

    downloadTrain() {
      window.location.href = `/v1/export/training/${this.exportId}/train.jsonl`;
    },

    downloadValid() {
      window.location.href = `/v1/export/training/${this.exportId}/valid.jsonl`;
    },
  };
}
