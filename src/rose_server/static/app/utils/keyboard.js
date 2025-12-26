const REGEX_DELIMITER = /[.,!?;:)]/;

export function isDelimiterKey(key) {
  return key === " " || REGEX_DELIMITER.test(key);
}
