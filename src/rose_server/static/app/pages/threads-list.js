export const threadsListPage = () => ({});

export const deleteConfirmPopover = (threadId) => ({
  open() {
    this.$store.threads.setCurrentThread(threadId);
    this.$refs.popover.showPopover();
  },

  close() {
    this.$refs.popover.hidePopover();
    this.$store.threads.clearCurrentThread();
  },

  get isDeleting() {
    return this.$store.threads.isDeleting(threadId);
  },

  async confirmDelete() {
    this.$store.threads.setDeleting(true);

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
      this.$store.threads.setDeleting(false);
    }
  },
});
