export function computeMentionState({ controller, lensOptions, tagOptions, previous }) {
  if (!controller) {
    throw new Error("controller is required");
  }

  const mention = controller.detectMention();
  if (!mention) {
    return {
      open: false,
      query: "",
      options: lensOptions,
      index: 0,
      mention: null,
    };
  }

  const query = mention.query;
  const options =
    mention.type === "tag"
      ? tagOptions.filter((option) => option.tag.startsWith(query))
      : lensOptions.filter((option) => option.atName.startsWith(query));

  const wasOpen = Boolean(previous?.open) && previous?.query === query;
  const open = options.length > 0;
  const maxIndex = Math.max(options.length - 1, 0);

  return {
    open,
    query,
    options,
    index: wasOpen ? Math.min(previous?.index ?? 0, maxIndex) : 0,
    mention,
  };
}
