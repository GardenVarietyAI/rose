import { claudeCodeValidator } from "../utils/import-validators/claude-code.js";

const VALIDATORS = {
  "claude-code": claudeCodeValidator,
};

export const importPage = () => ({
  preview: false,
  importing: false,
  complete: false,
  threads: [],
  parseReport: null,
  selectedFormat: "claude-code",
  banner: {
    visible: false,
    type: "",
    message: "",
    timeout: null
  },

  get currentValidator() {
    return VALIDATORS[this.selectedFormat];
  },

  handleFormatChange() {
    this.threads = [];
    this.parseReport = null;
    this.preview = false;
    this.closeBanner();
    if (this.$refs.fileInput) {
      this.$refs.fileInput.value = "";
    }
  },

  showBanner(type, message) {
    this.banner.visible = true;
    this.banner.type = type;
    this.banner.message = message;

    if (this.banner.timeout) {
      clearTimeout(this.banner.timeout);
    }

    this.banner.timeout = setTimeout(() => {
      this.banner.visible = false;
    }, 5000);
  },

  closeBanner() {
    this.banner.visible = false;
    if (this.banner.timeout) {
      clearTimeout(this.banner.timeout);
    }
  },

  get selectedCount() {
    return this.threads.filter((t) => t.selected).length;
  },

  get totalMessageCount() {
    return this.threads
      .filter((t) => t.selected)
      .reduce((sum, t) => sum + 1 + t.assistantMessages.length, 0);
  },

  get parseReportSummary() {
    if (!this.parseReport) return "";
    const r = this.parseReport;
    const parts = [];

    parts.push(`${r.parsed} records parsed`);

    if (r.skipped > 0) {
      const reasons = [];
      if (r.errors.invalidJSON.length > 0) reasons.push(`${r.errors.invalidJSON.length} invalid JSON`);
      if (r.errors.invalidType > 0) reasons.push(`${r.errors.invalidType} invalid type`);
      if (r.errors.emptyContent > 0) reasons.push(`${r.errors.emptyContent} empty content`);
      if (r.errors.invalidTimestamp.length > 0) reasons.push(`${r.errors.invalidTimestamp.length} invalid timestamp`);
      if (r.errors.validationErrors.length > 0) reasons.push(`${r.errors.validationErrors.length} validation errors`);

      parts.push(`${r.skipped} skipped: ${reasons.join(", ")}`);
    }

    return parts.join(", ");
  },

  handleFileSelect(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target.result;
        const { threads, report } = this.currentValidator.parse(text);
        this.threads = threads;
        this.parseReport = report;
        this.preview = true;

        if (report.skipped > 0) {
          this.showBanner("error", this.parseReportSummary);
        }
      } catch (error) {
        console.error("Parse error:", error);
        this.showBanner("error", `Failed to parse file: ${error.message}`);
      }
    };
    reader.readAsText(file);
  },

  toggleAll() {
    const allSelected = this.threads.every((t) => t.selected);
    this.threads.forEach((t) => (t.selected = !allSelected));
  },

  cancelImport() {
    this.preview = false;
    this.threads = [];
    this.parseReport = null;
    this.closeBanner();
    this.$refs.fileInput.value = "";
  },

  async executeImport() {
    if (this.importing || this.selectedCount === 0) return;

    this.importing = true;
    try {
      const messages = [];
      for (const thread of this.threads.filter((t) => t.selected)) {
        messages.push({
          uuid: thread.userMessage.uuid,
          thread_id: thread.threadId,
          role: thread.userMessage.role,
          content: thread.userMessage.content,
          model: thread.userMessage.model,
          created_at: thread.userMessage.created_at,
          meta: {
            session_id: thread.userMessage.sessionId,
            original_timestamp: thread.userMessage.timestamp,
          },
        });

        for (const assistantMsg of thread.assistantMessages) {
          messages.push({
            uuid: assistantMsg.uuid,
            thread_id: thread.threadId,
            role: assistantMsg.role,
            content: assistantMsg.content,
            model: assistantMsg.model,
            created_at: assistantMsg.created_at,
            meta: {
              session_id: assistantMsg.sessionId,
              original_timestamp: assistantMsg.timestamp,
            },
          });
        }
      }

      const response = await fetch("/v1/import/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          import_source: this.currentValidator.importSource,
          messages,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const result = await response.json();
      this.preview = false;
      this.complete = true;
      this.showBanner("success", `Successfully imported ${result.imported} messages`);
    } catch (error) {
      console.error("Import error:", error);
      this.showBanner("error", `Failed to import: ${error.message}`);
      this.importing = false;
    }
  },
});
