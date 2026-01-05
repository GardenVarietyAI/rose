export function buildLensState(lensMap) {
  const lensOptions = Object.entries(lensMap || {})
    .map(([atName, lensId]) => ({ lensId, atName: String(atName || "").toLowerCase() }))
    .filter((option) => option.atName);

  const idToAtName = Object.fromEntries(
    Object.entries(lensMap || {}).map(([atName, lensId]) => [lensId, atName])
  );

  return { lensOptions, idToAtName };
}

export function buildFactsheetState(factsheetMap, factsheetTitleMap) {
  const tagToFactsheetId = {};
  const tagToTitle = {};
  const factsheetIdToTag = {};
  const factsheetIdToTitle = {};
  const options = [];

  for (const [tag, factsheetId] of Object.entries(factsheetMap || {})) {
    if (!tag || !factsheetId) continue;
    const normalizedTag = String(tag).toLowerCase();
    tagToFactsheetId[normalizedTag] = factsheetId;
    const title = factsheetTitleMap?.[tag] || factsheetTitleMap?.[normalizedTag] || "";
    tagToTitle[normalizedTag] = title;
    factsheetIdToTag[factsheetId] = normalizedTag;
    factsheetIdToTitle[factsheetId] = title;
    options.push({ tag: normalizedTag, factsheetId, title: tagToTitle[normalizedTag] });
  }

  return {
    tagOptions: options.sort((a, b) => a.tag.localeCompare(b.tag)),
    tagToFactsheetId,
    tagToTitle,
    factsheetIdToTag,
    factsheetIdToTitle,
  };
}
