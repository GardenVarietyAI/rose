export const threadPage = () => ({
  pending: false,
  assistantEventSource: null,
  systemEventSource: null,

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

  cleanupEventStreams() {
    if (this.assistantEventSource) this.assistantEventSource.close();
    if (this.systemEventSource) this.systemEventSource.close();
    this.assistantEventSource = null;
    this.systemEventSource = null;
  },

  cleanupAssistantStream() {
    if (this.assistantEventSource) this.assistantEventSource.close();
    this.assistantEventSource = null;
  },

  cleanupSystemStream() {
    if (this.systemEventSource) this.systemEventSource.close();
    this.systemEventSource = null;
  },

  startEventStreams(threadId, tempId, { afterAssistantUuid = null, afterSystemUuid = null } = {}) {
    if (this.assistantEventSource || this.systemEventSource) return;

    const assistantUrl = new URL(`/v1/threads/${threadId}/events`, window.location.origin);
    assistantUrl.searchParams.set("role", "assistant");
    if (afterAssistantUuid) assistantUrl.searchParams.set("after_uuid", afterAssistantUuid);

    const systemUrl = new URL(`/v1/threads/${threadId}/events`, window.location.origin);
    systemUrl.searchParams.set("role", "system");
    if (afterSystemUuid) systemUrl.searchParams.set("after_uuid", afterSystemUuid);

    this.assistantEventSource = new EventSource(assistantUrl);
    this.systemEventSource = new EventSource(systemUrl);

    this.assistantEventSource.addEventListener("assistant", async (event) => {
      try {
        const data = JSON.parse(event?.data || "{}");
        const uuid = data?.uuid;
        if (!uuid) return;

        const fragmentResponse = await fetch(`/v1/messages/${uuid}/fragment`, {
          headers: { Accept: "text/html" },
        });
        if (!fragmentResponse.ok) return;

        const fragmentHtml = await fragmentResponse.text();
        const fragmentEl = this.parseHtmlElement(fragmentHtml);
        const placeholderEl = this.$refs.responses.querySelector(`[data-temp-id="${tempId}"]`);
        if (placeholderEl && fragmentEl) placeholderEl.replaceWith(fragmentEl);
      } finally {
        this.cleanupEventStreams();
      }
    });

    this.systemEventSource.addEventListener("system", async (event) => {
      const data = JSON.parse(event?.data || "{}");
      const uuid = data?.uuid;
      const meta = data?.meta || {};
      if (!uuid) return;

      if (meta?.object !== "job" || meta?.status !== "failed") return;

      try {
        const fragmentResponse = await fetch(`/v1/messages/${uuid}/fragment`, {
          headers: { Accept: "text/html" },
        });
        if (!fragmentResponse.ok) return;

        const fragmentHtml = await fragmentResponse.text();
        const fragmentEl = this.parseHtmlElement(fragmentHtml);
        const placeholderEl = this.$refs.responses.querySelector(`[data-temp-id="${tempId}"]`);
        if (placeholderEl && fragmentEl) placeholderEl.replaceWith(fragmentEl);
      } finally {
        this.cleanupEventStreams();
      }
    });

    this.assistantEventSource.addEventListener("error", () => this.cleanupAssistantStream());
    this.systemEventSource.addEventListener("error", () => this.cleanupSystemStream());
  },

  init() {
    if (!this.$refs.responses || this.$refs.responses.children.length) return;
    const threadId = this.$el?.dataset?.threadId;
    if (!threadId) return;

    const tempId = "initial";
    const placeholder = this.createPlaceholder(tempId);
    this.$refs.responses.prepend(placeholder);
    this.startEventStreams(threadId, tempId);
  },

  async regenerate(event) {
    if (this.pending) return;

    const link = event?.currentTarget;
    const threadId = link?.dataset?.threadId;
    const model = link?.dataset?.model;
    if (!threadId) return;

    this.pending = true;

    const tempId = Math.random().toString(36).slice(2, 10);
    const placeholder = this.createPlaceholder(tempId);
    this.$refs.responses.prepend(placeholder);

    try {
      const latestAssistant = this.$refs.responses.querySelector(".message[data-uuid]");
      const afterAssistantUuid = latestAssistant?.dataset?.uuid || null;

      const createMessageResponse = await fetch("/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          thread_id: threadId,
          model: model,
          generate_assistant: true,
          lens_id: this.$refs?.lensSelect?.value || undefined,
        }),
      });
      if (!createMessageResponse.ok) throw new Error(`Create message HTTP ${createMessageResponse.status}`);

      this.startEventStreams(threadId, tempId, { afterAssistantUuid });
    } catch (error) {
      console.error("Error:", error);
      const placeholderEl = this.$refs.responses.querySelector(`[data-temp-id="${tempId}"]`);
      if (placeholderEl) placeholderEl.remove();
    } finally {
      this.pending = false;
    }
  },
});
