export function importPage() {
  return {
    file: null,
    importSource: "",
    conversations: [],
    importing: false,
    stats: null,

    get totalMessages() {
      return this.conversations.reduce(
        (sum, conv) => sum + conv.messages.length,
        0
      );
    },

    async handleFile(event) {
      this.file = event.target.files[0];
      if (!this.file) {
        this.conversations = [];
        return;
      }

      try {
        const text = await this.file.text();
        const lines = text.split("\n").filter((line) => line.trim());

        this.conversations = lines.map((line) => JSON.parse(line));
      } catch (error) {
        console.error("Parse error:", error);
        alert("Failed to parse JSONL file. Please check the format.");
        this.conversations = [];
      }
    },

    async submitImport() {
      if (!this.file || !this.importSource) {
        return;
      }

      this.importing = true;
      try {
        const response = await fetch("/v1/import", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            import_source: this.importSource,
            conversations: this.conversations,
          }),
        });

        if (!response.ok) {
          throw new Error("Import failed");
        }

        this.stats = await response.json();
      } catch (error) {
        console.error("Import error:", error);
        alert("Import failed. Please try again.");
      } finally {
        this.importing = false;
      }
    },
  };
}
