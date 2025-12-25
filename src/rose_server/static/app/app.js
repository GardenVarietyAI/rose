import { askButton } from "./components/ask-button.js";
import { responseMessage } from "./components/response-message.js";
import { searchForm } from "./components/search-form.js";
import { threadPage } from "./pages/thread.js";

document.addEventListener("alpine:init", () => {
  Alpine.store("search", {
    query: "",
    lensId: "",
    exact: false,
    limit: 10,
  });
  Alpine.data("askButton", askButton);
  Alpine.data("responseMessage", responseMessage);
  Alpine.data("searchForm", searchForm);
  Alpine.data("threadPage", threadPage);
});
