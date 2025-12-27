import * as v from '../../../vendor/valibot/valibot.min.js';

const MessageSchema = v.object({
  role: v.string(),
  content: v.optional(v.union([
    v.string(),
    v.array(v.object({
      type: v.optional(v.string()),
      text: v.optional(v.string())
    }))
  ])),
  model: v.optional(v.string())
});

const ClaudeCodeRecordSchema = v.object({
  uuid: v.string(),
  type: v.union([v.literal("user"), v.literal("assistant")]),
  timestamp: v.pipe(v.string(), v.isoTimestamp()),
  sessionId: v.optional(v.string()),
  message: MessageSchema
});

function validateRecord(record, lineNum) {
  const result = v.safeParse(ClaudeCodeRecordSchema, record);

  if (!result.success) {
    return result.issues.map(issue => {
      const path = issue.path?.map(p => p.key).join('.') || 'record';
      return `Line ${lineNum}: ${path} - ${issue.message}`;
    });
  }

  return [];
}

function parseClaudeCodeJSONL(text) {
  const lines = text.trim().split("\n");
  const sessionMessages = new Map();
  const MAX_ERROR_EXAMPLES = 50;
  const parseReport = {
    totalLines: lines.length,
    parsed: 0,
    skipped: 0,
    errors: {
      invalidJSON: [],
      invalidType: 0,
      missingMessage: 0,
      missingRole: 0,
      emptyContent: 0,
      invalidTimestamp: [],
      validationErrors: []
    }
  };

  lines.forEach((line, idx) => {
    const lineNum = idx + 1;

    if (!line.trim()) return;

    let record;
    try {
      record = JSON.parse(line);
    } catch (e) {
      if (parseReport.errors.invalidJSON.length < MAX_ERROR_EXAMPLES) {
        parseReport.errors.invalidJSON.push(`Line ${lineNum}: ${e.message}`);
      }
      parseReport.skipped++;
      return;
    }

    const validationErrors = validateRecord(record, lineNum);
    if (validationErrors.length > 0) {
      const remaining = MAX_ERROR_EXAMPLES - parseReport.errors.validationErrors.length;
      if (remaining > 0) {
        parseReport.errors.validationErrors.push(...validationErrors.slice(0, remaining));
      }
      parseReport.skipped++;
      return;
    }

    if (record.type !== "user" && record.type !== "assistant") {
      parseReport.errors.invalidType++;
      parseReport.skipped++;
      return;
    }

    const msg = record.message;
    if (!msg || !msg.role) {
      parseReport.errors.missingRole++;
      parseReport.skipped++;
      return;
    }

    const content = Array.isArray(msg.content)
      ? msg.content.map((c) => (c.type === "text" ? c.text : "")).join("\n")
      : msg.content;

    if (!content || !content.trim()) {
      parseReport.errors.emptyContent++;
      parseReport.skipped++;
      return;
    }

    const sessionId = record.sessionId || `generated_${record.uuid}`;

    if (!sessionMessages.has(sessionId)) {
      sessionMessages.set(sessionId, []);
    }

    const timestamp = new Date(record.timestamp);

    sessionMessages.get(sessionId).push({
      uuid: record.uuid,
      role: msg.role,
      content: content.trim(),
      model: msg.model || null,
      created_at: Math.floor(timestamp.getTime() / 1000),
      timestamp: record.timestamp,
      sessionId: sessionId,
    });

    parseReport.parsed++;
  });

  const threads = [];
  for (const messages of sessionMessages.values()) {
    let currentThread = null;

    for (const msg of messages) {
      if (msg.role === "user") {
        if (currentThread && currentThread.length > 0) {
          threads.push(currentThread);
        }
        currentThread = [msg];
      } else if (msg.role === "assistant" && currentThread) {
        currentThread.push(msg);
      }
    }

    if (currentThread && currentThread.length > 0) {
      threads.push(currentThread);
    }
  }

  const threadObjects = threads.map((thread, idx) => ({
    id: idx,
    threadId: `import_${thread[0].sessionId}_${thread[0].uuid}`,
    userMessage: thread[0],
    assistantMessages: thread.slice(1),
    selected: false,
  }));

  return { threads: threadObjects, report: parseReport };
}

export const claudeCodeValidator = {
  name: "Claude Code JSONL",
  fileExtension: ".jsonl",
  parse: parseClaudeCodeJSONL
};
