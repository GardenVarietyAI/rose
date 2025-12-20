import { askButton } from "./components/ask-button.js";
import { responseMessage } from "./components/response-message.js";
import { threadPage } from "./pages/thread.js";

document.addEventListener("alpine:init", () => {
  Alpine.data("askButton", askButton);
  Alpine.data("responseMessage", responseMessage);
  Alpine.data("threadPage", threadPage);
});
