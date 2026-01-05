import { askButton } from "./components/ask-button.js";
import { responseMessage } from "./components/response-message.js";
import { searchForm } from "./components/search-form.js";
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

    applyQueryModel(model) {
      const parsed = parseQueryModel(model);

      this.content = parsed.content;
      this.lens_ids = parsed.lens_ids;
      this.factsheet_ids = parsed.factsheet_ids;
      this.exact = parsed.exact;
      this.limit = parsed.limit;
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
  Alpine.data("importPage", importPage);
  Alpine.data("threadMessagesPage", threadMessagesPage);
  Alpine.data("threadsListPage", threadsListPage);
  Alpine.data("deleteConfirmPopover", deleteConfirmPopover);
});
