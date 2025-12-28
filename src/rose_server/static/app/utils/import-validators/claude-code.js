import * as v from '../../../vendor/valibot/valibot.min.js';

const ROLE_SCHEMA = v.union([v.literal("user"), v.literal("assistant")]);

const MessageContentBlockSchema = v.object({
  type: v.optional(v.string()),
  text: v.optional(v.string()),
});

const MessageContentSchema = v.optional(v.union([v.string(), v.array(MessageContentBlockSchema)]));

const MessageSchema = v.object({
  role: ROLE_SCHEMA,
  content: MessageContentSchema,
  model: v.optional(v.string()),
});

const ClaudeCodeRecordSchema = v.object({
  uuid: v.string(),
  type: ROLE_SCHEMA,
  timestamp: v.pipe(v.string(), v.isoTimestamp()),
  sessionId: v.optional(v.string()),
  message: MessageSchema
});

function formatValibotIssues(issues, lineNum) {
  return issues.map((issue) => {
    const path = issue.path?.map((p) => p.key).join('.') || 'record';
    return `Line ${lineNum}: ${path} - ${issue.message}`;
  });
}

function normalizeMessageContent(content) {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((block) => (block.type === "text" && typeof block.text === "string" ? block.text : ""))
      .join("\n");
  }

  return "";
}

function parseClaudeCodeJSONL(text) {
  const lines = text.split(/\r?\n/);
  const sessionMessages = new Map();
  const MAX_ERROR_EXAMPLES = 50;
  const parseReport = {
    totalLines: lines.length,
    parsed: 0,
    skipped: 0,
    errors: {
      invalidJSON: [],
      emptyContent: 0,
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

    const parsed = v.safeParse(ClaudeCodeRecordSchema, record);
    if (!parsed.success) {
      const remaining = MAX_ERROR_EXAMPLES - parseReport.errors.validationErrors.length;
      if (remaining > 0) {
        parseReport.errors.validationErrors.push(
          ...formatValibotIssues(parsed.issues, lineNum).slice(0, remaining)
        );
      }
      parseReport.skipped++;
      return;
    }

    const normalized = parsed.output;

    if (normalized.message.role !== normalized.type) {
      if (parseReport.errors.validationErrors.length < MAX_ERROR_EXAMPLES) {
        parseReport.errors.validationErrors.push(
          `Line ${lineNum}: message.role does not match record.type`
        );
      }
      parseReport.skipped++;
      return;
    }

    const content = normalizeMessageContent(normalized.message.content).trim();
    if (!content) {
      parseReport.errors.emptyContent++;
      parseReport.skipped++;
      return;
    }

    const sessionId = normalized.sessionId || `generated_${normalized.uuid}`;

    if (!sessionMessages.has(sessionId)) {
      sessionMessages.set(sessionId, []);
    }

    const timestamp = new Date(normalized.timestamp);

    sessionMessages.get(sessionId).push({
      uuid: normalized.uuid,
      role: normalized.message.role,
      content,
      model: normalized.message.model || null,
      created_at: Math.floor(timestamp.getTime() / 1000),
      timestamp: normalized.timestamp,
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
    threadId: thread[0].uuid,
    userMessage: thread[0],
    assistantMessages: thread.slice(1),
    selected: false,
  }));

  return { threads: threadObjects, report: parseReport };
}

export const claudeCodeValidator = {
  name: "Claude Code JSONL",
  fileExtension: ".jsonl",
  importSource: "claude_code_jsonl",
  parse: parseClaudeCodeJSONL
};
