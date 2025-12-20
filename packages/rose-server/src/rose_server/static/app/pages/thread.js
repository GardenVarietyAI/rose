const threadPage = () => ({
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
          model: model,
          messages: [{ role: "user", content: promptContent }],
        }),
      });
      if (!completionResponse.ok) throw new Error(`Completion HTTP ${completionResponse.status}`);

      const data = await completionResponse.json();
      const uuid = data.message_uuid;
      if (!uuid) throw new Error("Chat response missing message uuid");

      const fragmentResponse = await fetch(`/v1/messages/${uuid}/fragment`, {
        headers: { Accept: "text/html" },
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
  },
});

document.addEventListener("alpine:init", () => {
  Alpine.data("threadPage", threadPage);
});
