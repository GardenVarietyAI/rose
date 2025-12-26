export const threadMessagesPage = () => ({
  pending: false,
  assistantEventSource: null,
  jobPollTimeout: null,
  jobPollAttempt: 0,

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

  cleanupAssistantStream() {
    if (this.assistantEventSource) this.assistantEventSource.close();
    this.assistantEventSource = null;
  },

  cleanupJobPoll() {
    if (this.jobPollTimeout) clearTimeout(this.jobPollTimeout);
    this.jobPollTimeout = null;
    this.jobPollAttempt = 0;
  },

  async pollJobStatus(threadId, tempId) {
    try {
      const response = await fetch(`/v1/threads/${threadId}/activity`, {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) return;

      const data = await response.json();
      const latestEvent = data?.job_events?.[0];
      if (latestEvent?.status === "failed") {
        const placeholderEl = this.$refs.responses.querySelector(`[data-temp-id="${tempId}"]`);
        if (placeholderEl) placeholderEl.remove();
        this.cleanupJobPoll();
        return;
      }

      this.jobPollAttempt += 1;
      const delay = Math.min(1000 * Math.pow(2, this.jobPollAttempt), 30000);
      this.jobPollTimeout = setTimeout(() => this.pollJobStatus(threadId, tempId), delay);
    } catch (error) {
      console.error("Error polling job status:", error);
      this.jobPollAttempt += 1;
      const delay = Math.min(1000 * Math.pow(2, this.jobPollAttempt), 30000);
      this.jobPollTimeout = setTimeout(() => this.pollJobStatus(threadId, tempId), delay);
    }
  },

  startEventStreams(threadId, tempId, { afterAssistantUuid = null } = {}) {
    if (this.assistantEventSource) return;

    const assistantUrl = new URL(`/v1/threads/${threadId}/stream`, window.location.origin);
    assistantUrl.searchParams.set("role", "assistant");
    if (afterAssistantUuid) assistantUrl.searchParams.set("after_uuid", afterAssistantUuid);

    this.assistantEventSource = new EventSource(assistantUrl);

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
        this.cleanupJobPoll();
      } finally {
        this.cleanupAssistantStream();
      }
    });

    this.assistantEventSource.addEventListener("error", () => this.cleanupAssistantStream());

    this.jobPollAttempt = 0;
    this.pollJobStatus(threadId, tempId);
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
    const promptContent = link?.dataset?.promptContent;
    if (!threadId || !promptContent) return;

    this.pending = true;

    const tempId = Math.random().toString(36).slice(2, 10);
    const placeholder = this.createPlaceholder(tempId);
    this.$refs.responses.prepend(placeholder);

    try {
      const latestAssistant = this.$refs.responses.querySelector(".message[data-uuid]");
      const afterAssistantUuid = latestAssistant?.dataset?.uuid || null;

      const askResponse = await fetch("/v1/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: promptContent,
          thread_id: threadId,
          lens_id: this.$refs?.lensSelect?.value || undefined,
          model: model,
        }),
      });
      if (!askResponse.ok) throw new Error(`Ask HTTP ${askResponse.status}`);

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
