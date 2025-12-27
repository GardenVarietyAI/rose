export const threadsListPage = () => ({
  deleting: null,

  async deleteThread(threadId) {
    if (!confirm("Delete this thread? This will soft-delete all messages in the thread.")) {
      return;
    }

    this.deleting = threadId;

    try {
      const response = await fetch(`/v1/threads/${threadId}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      window.location.reload();
    } catch (error) {
      console.error("Delete error:", error);
      alert(`Failed to delete thread: ${error.message}`);
      this.deleting = null;
    }
  },
});
