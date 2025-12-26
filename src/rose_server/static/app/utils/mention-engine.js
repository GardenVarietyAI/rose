import { TOKEN_TYPES, tokenize } from "./tokenizer.js";

const clampIndex = (value, min, max) => Math.min(Math.max(value, min), max);

export function createMentionEngine() {
  let state = {
    open: false,
    query: "",
    index: 0,
    options: [],
    matchedToken: "",
  };

  const resetState = () => {
    state = { open: false, query: "", index: 0, options: [], matchedToken: "" };
  };

  const snapshot = () => ({
    open: state.open,
    query: state.query,
    options: state.options,
    index: state.index,
  });

  return {
    update(textBeforeCaret, allOptions) {
      const tokens = tokenize(textBeforeCaret);
      if (!tokens.length) {
        resetState();
        return snapshot();
      }

      const lastToken = tokens[tokens.length - 1];
      if (lastToken.type !== TOKEN_TYPES.MENTION) {
        resetState();
        return snapshot();
      }

      const query = lastToken.value.toLowerCase();
      const options = (allOptions || []).filter((option) =>
        option.atName.startsWith(query)
      );
      const index = state.query === query
        ? clampIndex(state.index, 0, Math.max(options.length - 1, 0))
        : 0;

      state = {
        open: options.length > 0,
        query,
        options,
        index,
        matchedToken: lastToken.raw,
      };
      return snapshot();
    },

    navigate(direction) {
      if (!state.options.length) return state.index;
      const delta = direction === "up" ? -1 : 1;
      state.index = clampIndex(
        state.index + delta,
        0,
        Math.max(state.options.length - 1, 0)
      );
      return state.index;
    },

    select() {
      if (!state.options.length || !state.open) return null;
      return {
        option: state.options[state.index],
        matchedToken: state.matchedToken,
      };
    },

    reset() {
      resetState();
    },
  };
}
