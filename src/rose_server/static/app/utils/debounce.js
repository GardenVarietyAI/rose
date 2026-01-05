export function debounce(fn, delayMs) {
  if (typeof fn !== "function") {
    throw new Error("fn must be a function");
  }
  const delay = Number(delayMs);
  if (!Number.isFinite(delay) || delay < 0) {
    throw new Error("delayMs must be a non-negative number");
  }

  let timeoutId = null;

  return (...args) => {
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      timeoutId = null;
      fn(...args);
    }, delay);
  };
}
