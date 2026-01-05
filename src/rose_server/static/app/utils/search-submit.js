function encodeSq(params) {
  const raw = JSON.stringify(params);
  return btoa(unescape(encodeURIComponent(raw)))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
}

function getQueryDefaults() {
  const transport = window.TRANSPORT?.search;
  return {
    exact: Boolean(transport?.exact),
    limit: typeof transport?.limit === "number" ? transport.limit : 10,
  };
}

export async function submitSearchFragment({
  payload,
  formAction,
  rootSelector = "#search-root",
  updateUrl,
}) {
  if (!payload) {
    throw new Error("payload is required");
  }
  if (!formAction) {
    throw new Error("formAction is required");
  }

  const response = await fetch("/v1/search/fragment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Search failed (HTTP ${response.status})`);
  }

  const html = await response.text();
  const current = document.querySelector(rootSelector);
  if (!current) {
    throw new Error(`Page missing ${rootSelector} element`);
  }
  current.outerHTML = html;

  const defaults = getQueryDefaults();
  const shouldIncludeSq =
    payload.lens_ids.length > 0 ||
    payload.factsheet_ids.length > 0 ||
    payload.exact !== defaults.exact ||
    payload.limit !== defaults.limit;

  const params = updateUrl === "sq"
    ? new URLSearchParams(window.location.search)
    : new URLSearchParams();

  if (!updateUrl) {
    if (payload.content) params.set("q", payload.content);
    else params.delete("q");
  }

  if (shouldIncludeSq) {
    params.set(
      "sq",
      encodeSq({
        lens_ids: payload.lens_ids,
        factsheet_ids: payload.factsheet_ids,
        exact: payload.exact,
        limit: payload.limit,
      })
    );
  } else {
    params.delete("sq");
  }

  const url = params.toString() ? `${formAction}?${params.toString()}` : formAction;
  window.history.replaceState({}, "", url);
}
