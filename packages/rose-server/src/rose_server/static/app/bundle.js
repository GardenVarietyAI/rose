(() => {
  // packages/rose-server/src/rose_server/static/app/components/ask-button.js
  var askButton = () => ({
    disabled: false,
    label: "Ask",
    interval: null,
    dots: 0,
    stopSpinner() {
      if (this.interval) clearInterval(this.interval);
      this.interval = null;
      this.dots = 0;
      this.label = "Ask";
    },
    startSpinner() {
      this.stopSpinner();
      this.interval = setInterval(() => {
        this.dots = this.dots % 3 + 1;
        this.label = "Asking" + ".".repeat(this.dots);
      }, 400);
    },
    getQuery() {
      const form = this.$el.closest("form");
      const queryInput = form?.querySelector('input[name="q"]');
      return queryInput?.value?.trim() || "";
    },
    async ask() {
      const query = this.getQuery();
      if (!query || this.disabled) return;
      this.disabled = true;
      this.startSpinner();
      try {
        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: [{ role: "user", content: query }] })
        });
        const data = await response.json();
        if (!data?.thread_id) throw new Error("Missing thread_id");
        window.location.href = "/v1/threads/" + data.thread_id;
      } catch (error) {
        alert("Error: " + (error?.message || String(error)));
      } finally {
        this.stopSpinner();
        this.disabled = false;
      }
    }
  });

  // packages/rose-server/src/rose_server/static/app/components/response-message.js
  var responseMessage = () => ({
    uuid: null,
    accepted: false,
    init() {
      this.uuid = this.$el.dataset.uuid;
      this.accepted = this.$el.classList.contains("accepted");
    },
    async toggleAccepted() {
      try {
        const response = await fetch(`/v1/messages/${this.uuid}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ accepted: !this.accepted })
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        this.accepted = !this.accepted;
      } catch (error) {
        console.error("Error:", error);
      }
    }
  });

  // packages/rose-server/src/rose_server/static/app/pages/thread.js
  var threadPage = () => ({
    pending: false,
    createPlaceholder(tempId) {
      const clone = this.$refs.placeholderTemplate.content.cloneNode(true);
      const message = clone.querySelector(".message");
      message.dataset.tempId = tempId;
      return clone;
    },
    parseHtmlElement(html) {
      const template = document.createElement("template");
      template.innerHTML = String(html).trim();
      return template.content.firstElementChild;
    },
    async regenerate(event) {
      if (this.pending) return;
      const link = event?.currentTarget;
      const threadId = link?.dataset?.threadId;
      const promptContent = link?.dataset?.promptContent;
      const model = link?.dataset?.model;
      if (!threadId || !promptContent) return;
      this.pending = true;
      const tempId = Math.random().toString(36).slice(2, 10);
      const placeholder = this.createPlaceholder(tempId);
      this.$refs.responses.prepend(placeholder);
      try {
        const completionResponse = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            thread_id: threadId,
            model,
            messages: [{ role: "user", content: promptContent }]
          })
        });
        if (!completionResponse.ok) throw new Error(`Completion HTTP ${completionResponse.status}`);
        const data = await completionResponse.json();
        const uuid = data.message_uuid;
        if (!uuid) throw new Error("Chat response missing message uuid");
        const fragmentResponse = await fetch(`/v1/messages/${uuid}/fragment`, {
          headers: { Accept: "text/html" }
        });
        if (!fragmentResponse.ok) throw new Error(`Fragment HTTP ${fragmentResponse.status}`);
        const fragmentHtml = await fragmentResponse.text();
        const fragmentEl = this.parseHtmlElement(fragmentHtml);
        if (!fragmentEl) throw new Error("Empty fragment");
        const placeholderEl = this.$refs.responses.querySelector(`[data-temp-id="${tempId}"]`);
        if (placeholderEl) placeholderEl.replaceWith(fragmentEl);
      } catch (error) {
        console.error("Error:", error);
        const placeholderEl = this.$refs.responses.querySelector(`[data-temp-id="${tempId}"]`);
        if (placeholderEl) placeholderEl.remove();
      } finally {
        this.pending = false;
      }
    }
  });

  // packages/rose-server/src/rose_server/static/app/app.js
  document.addEventListener("alpine:init", () => {
    Alpine.data("askButton", askButton);
    Alpine.data("responseMessage", responseMessage);
    Alpine.data("threadPage", threadPage);
  });
})();
