import { sanitizeText } from "../../utils/text.js";

export const lensToken = (lensId, rawAtName) => ({
  lensId,
  rawAtName,

  init() {
    if (!this.lensId || !this.rawAtName) {
      throw new Error("lensId and rawAtName are required");
    }
    this.render();
  },

  render() {
    const el = this.$el;
    const processedAtName = sanitizeText(this.rawAtName.toLowerCase());
    el.className = "query-token";
    el.dataset.token = "lens";
    el.dataset.lensId = this.lensId;
    el.dataset.atName = processedAtName;
    el.textContent = `@${processedAtName}`;
    el.setAttribute("contenteditable", "false");
    el.style.cursor = "pointer";
  },

  removeLens() {
    this.$store.search.clearLens();
  },
});
