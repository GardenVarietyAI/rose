-- migrate:up
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL DEFAULT 'thread.run',
    created_at INTEGER NOT NULL,
    thread_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,

    -- Status fields
    status TEXT NOT NULL DEFAULT 'queued',
    started_at INTEGER,
    expires_at INTEGER,
    failed_at INTEGER,
    completed_at INTEGER,
    cancelled_at INTEGER,

    -- Configuration
    model TEXT NOT NULL,
    instructions TEXT,
    additional_instructions TEXT,
    additional_messages TEXT,  -- JSON, nullable
    tools TEXT NOT NULL DEFAULT '[]',  -- JSON
    meta TEXT NOT NULL DEFAULT '{}',  -- JSON

    -- Model parameters
    temperature REAL,
    top_p REAL,
    max_prompt_tokens INTEGER,
    max_completion_tokens INTEGER,

    -- Tool configuration
    tool_choice TEXT,  -- JSON, nullable
    parallel_tool_calls INTEGER NOT NULL DEFAULT 1,  -- Boolean as integer

    -- Response configuration
    response_format TEXT,  -- JSON, nullable
    truncation_strategy TEXT,  -- JSON, nullable

    -- Results
    last_error TEXT,  -- JSON, nullable
    incomplete_details TEXT,  -- JSON, nullable
    usage TEXT,  -- JSON, nullable
    required_action TEXT,  -- JSON, nullable

    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE,
    FOREIGN KEY (assistant_id) REFERENCES assistants(id) ON DELETE CASCADE
);

-- Create indexes for runs
CREATE INDEX idx_runs_thread ON runs(thread_id);
CREATE INDEX idx_runs_assistant ON runs(assistant_id);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_created ON runs(created_at);

-- Create run_steps table
CREATE TABLE run_steps (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL DEFAULT 'thread.run.step',
    created_at INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,

    -- Step type and details
    type TEXT NOT NULL,
    step_details TEXT NOT NULL,  -- JSON

    -- Status fields
    status TEXT NOT NULL DEFAULT 'in_progress',
    cancelled_at INTEGER,
    completed_at INTEGER,
    expired_at INTEGER,
    failed_at INTEGER,

    -- Error and usage
    last_error TEXT,  -- JSON, nullable
    usage TEXT,  -- JSON, nullable

    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

-- Create indexes for run_steps
CREATE INDEX idx_run_steps_run ON run_steps(run_id);
CREATE INDEX idx_run_steps_created ON run_steps(created_at);
CREATE INDEX idx_run_steps_status ON run_steps(status);

-- migrate:down
DROP INDEX IF EXISTS idx_run_steps_status;
DROP INDEX IF EXISTS idx_run_steps_created;
DROP INDEX IF EXISTS idx_run_steps_run;
DROP TABLE IF EXISTS run_steps;

DROP INDEX IF EXISTS idx_runs_created;
DROP INDEX IF EXISTS idx_runs_status;
DROP INDEX IF EXISTS idx_runs_assistant;
DROP INDEX IF EXISTS idx_runs_thread;
DROP TABLE runs;
