import { askButton } from "./components/ask-button.js";
import { responseMessage } from "./components/response-message.js";
import { searchForm } from "./components/search-form.js";
import { exportPage } from "./pages/export.js";
import { importPage } from "./pages/importer.js";
import { threadMessagesPage } from "./pages/thread_messages.js";
import { deleteConfirmPopover, threadsListPage } from "./pages/threads-list.js";
import { markdownToHtml } from "./utils/markdown.js";
import { parseQueryModel } from "./utils/query-model.js";

document.addEventListener("alpine:init", () => {
  if (!window.TRANSPORT?.search) {
    throw new Error("TRANSPORT.search not found");
  }

  Alpine.store("search", {
    ...window.TRANSPORT.search,

    _commit(next) {
      const parsed = parseQueryModel({
        content: next.content,
        lens_ids: next.lens_ids,
        factsheet_ids: next.factsheet_ids,
        exact: next.exact,
        limit: next.limit,
      });

      this.content = parsed.content;
      this.lens_ids = parsed.lens_ids;
      this.factsheet_ids = parsed.factsheet_ids;
      this.exact = parsed.exact;
      this.limit = parsed.limit;
    },

    applyQueryModel(model) {
      this._commit(model);
    },

    setContent(content) {
      this._commit({
        ...this,
        content: String(content ?? ""),
      });
    },

    setLensId(lensId) {
      const normalized = String(lensId ?? "").trim();
      this._commit({
        ...this,
        lens_ids: normalized ? [normalized] : [],
      });
    },

    clearLens() {
      this.setLensId("");
    },

    addFactsheetId(factsheetId) {
      const normalized = String(factsheetId ?? "").trim();
      if (!normalized) return;
      if (this.factsheet_ids.includes(normalized)) return;

      this._commit({
        ...this,
        factsheet_ids: [...this.factsheet_ids, normalized],
      });
    },

    removeFactsheetId(factsheetId) {
      const normalized = String(factsheetId ?? "").trim();
      if (!normalized) return;

      this._commit({
        ...this,
        factsheet_ids: this.factsheet_ids.filter((candidate) => candidate !== normalized),
      });
    },

    setFactsheetIds(ids) {
      const normalized = Array.from(
        new Set((ids ?? []).map((id) => String(id).trim()).filter(Boolean))
      );
      this._commit({
        ...this,
        factsheet_ids: normalized,
      });
    },

    setExact(exact) {
      this._commit({
        ...this,
        exact: Boolean(exact),
      });
    },

    setLimit(limit) {
      this._commit({
        ...this,
        limit: Number(limit),
      });
    },
  });

  Alpine.store("threads", {
    ...window.TRANSPORT.threads,

    setCurrentThread(threadId) {
      this.currentThreadId = threadId;
    },

    clearCurrentThread() {
      this.currentThreadId = null;
    },

    setDeleting(value) {
      this.deleting = value;
    },

    isDeleting(threadId) {
      return this.deleting && this.currentThreadId === threadId;
    },
  });

  Alpine.magic("markdown", () => markdownToHtml);
  Alpine.data("askButton", askButton);
  Alpine.data("responseMessage", responseMessage);
  Alpine.data("searchForm", searchForm);
  Alpine.data("exportPage", exportPage);
  Alpine.data("importPage", importPage);
  Alpine.data("threadMessagesPage", threadMessagesPage);
  Alpine.data("threadsListPage", threadsListPage);
  Alpine.data("deleteConfirmPopover", deleteConfirmPopover);
});
