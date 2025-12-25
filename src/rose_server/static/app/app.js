import { askButton } from "./components/ask-button.js";
import { responseMessage } from "./components/response-message.js";
import { searchForm } from "./components/search-form.js";
import { lensToken } from "./components/tokens/lens-token.js";
import { threadPage } from "./pages/thread.js";

document.addEventListener("alpine:init", () => {
  if (!window.TRANSPORT?.search) {
    throw new Error("TRANSPORT.search not found");
  }

  Alpine.store("search", {
    ...window.TRANSPORT.search,

    clearLens() {
      this.lens_id = "";
    },
  });

  Alpine.data("askButton", askButton);
  Alpine.data("responseMessage", responseMessage);
  Alpine.data("searchForm", searchForm);
  Alpine.data("lensToken", lensToken);
  Alpine.data("threadPage", threadPage);
});
