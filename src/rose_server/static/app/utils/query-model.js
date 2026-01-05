import validateQueryRequest from "./query-request.validate.js";

export function parseQueryModel(input) {
  if (!validateQueryRequest(input)) {
    const details = validateQueryRequest.errors
      ? JSON.stringify(validateQueryRequest.errors)
      : "unknown validation error";
    throw new Error(`Invalid query model: ${details}`);
  }

  return {
    content: input.content ?? "",
    lens_ids: input.lens_ids ?? [],
    factsheet_ids: input.factsheet_ids ?? [],
    exact: input.exact ?? false,
    limit: input.limit ?? 10,
  };
}
