function highlightCode(code, lang) {
  if (typeof code !== "string") {
    throw new TypeError("highlightCode expects a string");
  }

  const detectedLang = detectLanguage(code, lang);

  const strings = [];
  const comments = [];
  const keywords = [];
  const builtins = [];
  const numbers = [];

  const encodeIndex = (i) => {
    if (!Number.isInteger(i) || i < 0) {
      throw new TypeError("encodeIndex expects a non-negative integer");
    }

    const alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const base = alphabet.length;
    let n = i;
    let out = "";
    do {
      out = alphabet[n % base] + out;
      n = Math.floor(n / base);
    } while (n > 0);
    return out;
  };

  const stringToken = (i) => `\u0001HLSTR${encodeIndex(i)}\u0001`;
  const commentToken = (i) => `\u0001HLCMT${encodeIndex(i)}\u0001`;
  const keywordToken = (i) => `\u0001HLKW${encodeIndex(i)}\u0001`;
  const builtinToken = (i) => `\u0001HLBI${encodeIndex(i)}\u0001`;
  const numberToken = (i) => `\u0001HLNUM${encodeIndex(i)}\u0001`;

  const matchIndex = (encoded) => {
    const alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const base = alphabet.length;
    let n = 0;
    for (const ch of String(encoded)) {
      const idx = alphabet.indexOf(ch);
      if (idx === -1) {
        throw new Error("Invalid placeholder encoding");
      }
      n = (n * base) + idx;
    }
    return n;
  };

  let s = code;

  if (detectedLang === "python") {
    s = s.replace(/#[^\n]*/g, (match) => {
      const i = comments.length;
      comments.push(match);
      return commentToken(i);
    });
    s = s.replace(/'''[\s\S]*?'''|"""[\s\S]*?"""/g, (match) => {
      const i = strings.length;
      strings.push(match);
      return stringToken(i);
    });
    s = s.replace(/'([^'\\]|\\.)*?'|"([^"\\]|\\.)*?"/g, (match) => {
      const i = strings.length;
      strings.push(match);
      return stringToken(i);
    });
  } else if (detectedLang === "sql") {
    s = s.replace(/--[^\n]*/g, (match) => {
      const i = comments.length;
      comments.push(match);
      return commentToken(i);
    });
    s = s.replace(/'([^'\\]|\\.)*?'/g, (match) => {
      const i = strings.length;
      strings.push(match);
      return stringToken(i);
    });
  } else {
    s = s.replace(/\/\/[^\n]*/g, (match) => {
      const i = comments.length;
      comments.push(match);
      return commentToken(i);
    });
    s = s.replace(/\/\*[\s\S]*?\*\//g, (match) => {
      const i = comments.length;
      comments.push(match);
      return commentToken(i);
    });
    s = s.replace(/'([^'\\]|\\.)*?'|"([^"\\]|\\.)*?"|`([^`\\]|\\.)*?`/g, (match) => {
      const i = strings.length;
      strings.push(match);
      return stringToken(i);
    });
  }

  s = escapeHtml(s);

  const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  const keywordList = getKeywords(detectedLang);
  if (keywordList.length > 0) {
    const flags = detectedLang === "sql" ? "gi" : "g";
    const kwPattern = new RegExp(`\\b(${keywordList.map(escapeRegExp).join("|")})\\b`, flags);
    s = s.replace(kwPattern, (match) => {
      const i = keywords.length;
      keywords.push(match);
      return keywordToken(i);
    });
  }

  const builtinList = getBuiltins(detectedLang);
  if (builtinList.length > 0) {
    const flags = detectedLang === "sql" ? "gi" : "g";
    const biPattern = new RegExp(`\\b(${builtinList.map(escapeRegExp).join("|")})\\b`, flags);
    s = s.replace(biPattern, (match) => {
      const i = builtins.length;
      builtins.push(match);
      return builtinToken(i);
    });
  }

  s = s.replace(/\b\d+(\.\d+)?\b/g, (match) => {
    const i = numbers.length;
    numbers.push(match);
    return numberToken(i);
  });

  s = s.replace(/\u0001HLSTR([A-Za-z]+)\u0001/g, (match, encoded) => {
    const index = matchIndex(encoded);
    const str = strings[index];
    if (str === undefined) {
      throw new Error("Invalid string placeholder index");
    }
    return `<span class="hl-str">${escapeHtml(str)}</span>`;
  });

  s = s.replace(/\u0001HLCMT([A-Za-z]+)\u0001/g, (match, encoded) => {
    const index = matchIndex(encoded);
    const cmt = comments[index];
    if (cmt === undefined) {
      throw new Error("Invalid comment placeholder index");
    }
    return `<span class="hl-cmt">${escapeHtml(cmt)}</span>`;
  });

  s = s.replace(/\u0001HLNUM([A-Za-z]+)\u0001/g, (match, encoded) => {
    const index = matchIndex(encoded);
    const num = numbers[index];
    if (num === undefined) {
      throw new Error("Invalid number placeholder index");
    }
    return `<span class="hl-num">${num}</span>`;
  });

  s = s.replace(/\u0001HLBI([A-Za-z]+)\u0001/g, (match, encoded) => {
    const index = matchIndex(encoded);
    const bi = builtins[index];
    if (bi === undefined) {
      throw new Error("Invalid builtin placeholder index");
    }
    return `<span class="hl-bi">${bi}</span>`;
  });

  s = s.replace(/\u0001HLKW([A-Za-z]+)\u0001/g, (match, encoded) => {
    const index = matchIndex(encoded);
    const kw = keywords[index];
    if (kw === undefined) {
      throw new Error("Invalid keyword placeholder index");
    }
    return `<span class="hl-kw">${kw}</span>`;
  });

  return s;
}

function escapeHtml(s) {
  if (typeof s !== "string") {
    throw new TypeError("escapeHtml expects a string");
  }
  return s.replace(/[&<>"']/g, (c) => (
    c === "&" ? "&amp;" :
    c === "<" ? "&lt;" :
    c === ">" ? "&gt;" :
    c === '"' ? "&quot;" : "&apos;"
  ));
}

function detectLanguage(code, declaredLang) {
  if (declaredLang) {
    const normalized = declaredLang.toLowerCase();
    if (normalized === "js" || normalized === "javascript" || normalized === "typescript" || normalized === "ts") {
      return "javascript";
    }
    if (normalized === "py" || normalized === "python") {
      return "python";
    }
    if (normalized === "sql" || normalized === "postgres" || normalized === "postgresql" || normalized === "mysql") {
      return "sql";
    }
    if (normalized === "go" || normalized === "golang") {
      return "go";
    }
    if (normalized === "rs" || normalized === "rust") {
      return "rust";
    }
    if (normalized === "sh" || normalized === "bash" || normalized === "shell") {
      return "shell";
    }
    return normalized;
  }

  if (/\b(def|class)\s+\w+[\(:]/.test(code)) return "python";
  if (/\b(const|let|var|function)\s+\w+/.test(code)) return "javascript";
  if (/^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b/mi.test(code)) return "sql";
  if (/\bfn\s+\w+\(/.test(code)) return "rust";
  if (/\bfunc\s+\w+\(/.test(code)) return "go";

  return "generic";
}

function getKeywords(lang) {
  const keywords = {
    python: ["def", "class", "if", "else", "elif", "for", "while", "return", "import", "from", "as", "try", "except", "finally", "with", "lambda", "yield", "async", "await", "raise", "pass", "break", "continue", "in", "is", "not", "and", "or"],
    javascript: ["function", "const", "let", "var", "if", "else", "for", "while", "return", "import", "export", "default", "class", "extends", "async", "await", "yield", "typeof", "new", "delete", "this", "super", "static", "try", "catch", "finally", "throw", "break", "continue", "switch", "case"],
    sql: ["SELECT", "FROM", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "OUTER", "ON", "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE", "CREATE", "TABLE", "ALTER", "DROP", "INDEX", "ORDER", "BY", "GROUP", "HAVING", "LIMIT", "OFFSET", "AS", "AND", "OR", "NOT", "IN", "LIKE", "BETWEEN"],
    go: ["func", "package", "import", "var", "const", "type", "struct", "interface", "if", "else", "for", "range", "return", "defer", "go", "chan", "select", "case", "switch", "break", "continue"],
    rust: ["fn", "let", "mut", "const", "static", "struct", "enum", "impl", "trait", "type", "if", "else", "match", "for", "while", "loop", "return", "break", "continue", "use", "mod", "pub", "crate"],
    shell: ["if", "then", "else", "elif", "fi", "for", "while", "do", "done", "case", "esac", "function", "return", "exit", "export", "source"],
    generic: [],
  };
  return keywords[lang] || keywords.generic;
}

function getBuiltins(lang) {
  const builtins = {
    python: ["True", "False", "None", "self", "str", "int", "float", "bool", "list", "dict", "tuple", "set", "len", "range", "print", "open", "type", "isinstance", "super", "__init__", "__name__"],
    javascript: ["true", "false", "null", "undefined", "NaN", "Infinity", "console", "window", "document", "Array", "Object", "String", "Number", "Boolean", "Math", "Date", "Promise", "JSON"],
    sql: ["NULL", "TRUE", "FALSE", "COUNT", "SUM", "AVG", "MIN", "MAX", "DISTINCT", "EXISTS"],
    go: ["true", "false", "nil", "make", "new", "len", "cap", "append", "copy", "delete", "panic", "recover"],
    rust: ["true", "false", "None", "Some", "Ok", "Err", "Vec", "String", "Option", "Result", "println", "panic"],
    shell: ["echo", "cd", "ls", "pwd", "cat", "grep", "sed", "awk", "find", "chmod", "chown"],
    generic: ["true", "false", "null", "undefined"],
  };
  return builtins[lang] || builtins.generic;
}

export { highlightCode };
