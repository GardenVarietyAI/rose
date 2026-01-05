import {
  array,
  boolean,
  maxValue,
  minLength,
  minValue,
  number,
  object,
  parse,
  pipe,
  string
} from "../../vendor/valibot/valibot.min.js";

export const QueryModelSchema = object({
  content: string(),
  lens_ids: array(pipe(string(), minLength(1))),
  factsheet_ids: array(pipe(string(), minLength(1))),
  exact: boolean(),
  limit: pipe(number(), minValue(1), maxValue(100))
});

export function parseQueryModel(input) {
  return parse(QueryModelSchema, input);
}
