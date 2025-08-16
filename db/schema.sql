CREATE TABLE IF NOT EXISTS "schema_migrations" (version varchar(128) primary key);
CREATE TABLE assistants (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL DEFAULT 'assistant',
    created_at INTEGER NOT NULL,
    name TEXT,
    description TEXT,
    model TEXT NOT NULL,
    instructions TEXT,
    tools TEXT NOT NULL DEFAULT '[]',  -- JSON
    tool_resources TEXT NOT NULL DEFAULT '{}',  -- JSON
    meta TEXT NOT NULL DEFAULT '{}',  -- JSON
    temperature REAL DEFAULT 0.7,
    top_p REAL DEFAULT 0.8,
    response_format TEXT DEFAULT NULL  -- JSON
);
CREATE INDEX idx_assistants_created ON assistants(created_at);
CREATE INDEX idx_assistants_model ON assistants(model);
CREATE TABLE files (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL DEFAULT 'file',
    bytes INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    expires_at INTEGER,
    filename TEXT NOT NULL,
    purpose TEXT NOT NULL,
    status TEXT DEFAULT 'processed',
    status_details TEXT,
    storage_path TEXT NOT NULL
, content BLOB);
CREATE INDEX idx_files_created_at ON files(created_at);
CREATE INDEX idx_files_purpose ON files(purpose);
CREATE INDEX idx_files_status ON files(status);
CREATE TABLE threads (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL DEFAULT 'thread',
    created_at INTEGER NOT NULL,
    tool_resources TEXT,  -- JSON, nullable
    meta TEXT NOT NULL DEFAULT '{}'  -- JSON
);
CREATE INDEX idx_threads_created ON threads(created_at);
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL DEFAULT 'thread.message',
    created_at INTEGER NOT NULL,
    thread_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,  -- JSON
    assistant_id TEXT,
    run_id TEXT,
    attachments TEXT NOT NULL DEFAULT '[]',  -- JSON
    meta TEXT NOT NULL DEFAULT '{}',  -- JSON
    response_chain_id TEXT,
    status TEXT NOT NULL DEFAULT 'completed',
    incomplete_details TEXT,  -- JSON, nullable
    incomplete_at INTEGER,
    completed_at INTEGER,
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
);
CREATE INDEX idx_messages_thread ON messages(thread_id);
CREATE INDEX idx_messages_created ON messages(created_at);
CREATE INDEX idx_messages_role ON messages(role);
CREATE INDEX idx_messages_response_chain ON messages(response_chain_id);
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
CREATE INDEX idx_runs_thread ON runs(thread_id);
CREATE INDEX idx_runs_assistant ON runs(assistant_id);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_created ON runs(created_at);
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
CREATE INDEX idx_run_steps_run ON run_steps(run_id);
CREATE INDEX idx_run_steps_created ON run_steps(created_at);
CREATE INDEX idx_run_steps_status ON run_steps(status);
CREATE TABLE fine_tuning_jobs (
    id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,
    finished_at INTEGER,
    model TEXT NOT NULL,
    fine_tuned_model TEXT,
    organization_id TEXT NOT NULL DEFAULT 'org-local',
    result_files TEXT NOT NULL DEFAULT '[]',  -- JSON
    status TEXT NOT NULL,
    validation_file TEXT,
    training_file TEXT NOT NULL,
    error TEXT,  -- JSON, nullable
    seed INTEGER NOT NULL DEFAULT 42,
    trained_tokens INTEGER,
    meta TEXT,  -- JSON, nullable
    started_at INTEGER,
    suffix TEXT,
    hyperparameters TEXT NOT NULL DEFAULT '{}',  -- JSON
    method TEXT,  -- JSON, nullable
    trainer TEXT NOT NULL DEFAULT 'huggingface',
    training_metrics TEXT  -- JSON, nullable
);
CREATE INDEX idx_ft_jobs_status ON fine_tuning_jobs(status);
CREATE INDEX idx_ft_jobs_created ON fine_tuning_jobs(created_at);
CREATE TABLE fine_tuning_events (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL DEFAULT 'fine_tuning.job.event',
    created_at INTEGER NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    data TEXT,  -- JSON, nullable
    job_id TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES fine_tuning_jobs(id) ON DELETE CASCADE
);
CREATE INDEX idx_events_job_id ON fine_tuning_events(job_id);
CREATE INDEX idx_events_created ON fine_tuning_events(created_at);
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,  -- JSON
    result TEXT,  -- JSON, nullable
    error TEXT,
    created_at INTEGER NOT NULL,
    started_at INTEGER,
    completed_at INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3
);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_type_status ON jobs(type, status);
CREATE TABLE vector_stores (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL DEFAULT 'vector_store',
    name TEXT NOT NULL,
    dimensions INTEGER NOT NULL DEFAULT 384,
    meta TEXT NOT NULL DEFAULT '{}',  -- JSON for additional metadata
    created_at INTEGER NOT NULL
, last_used_at INTEGER);
CREATE INDEX idx_vector_stores_created_at ON vector_stores(created_at);
CREATE INDEX idx_vector_stores_name ON vector_stores(name);
CREATE TABLE message_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);
CREATE INDEX idx_message_metadata_message ON message_metadata(message_id);
CREATE INDEX idx_message_metadata_key ON message_metadata(key);
CREATE TABLE models (
    id TEXT PRIMARY KEY,
    name TEXT,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL DEFAULT 'huggingface',
    path TEXT,
    is_fine_tuned INTEGER NOT NULL DEFAULT 0,  -- Boolean as integer

    -- Model parameters
    temperature REAL NOT NULL DEFAULT 0.7,
    top_p REAL NOT NULL DEFAULT 0.9,
    memory_gb REAL NOT NULL DEFAULT 2.0,
    timeout INTEGER,
    quantization TEXT,

    -- LoRA configuration
    lora_target_modules TEXT,  -- JSON array

    -- OpenAI API compatibility fields
    owned_by TEXT NOT NULL DEFAULT 'organization-owner',
    parent TEXT,
    permissions TEXT DEFAULT '[]',  -- JSON array

    -- Metadata
    created_at INTEGER NOT NULL
);
CREATE INDEX idx_models_model_name ON models(model_name);
CREATE INDEX idx_models_is_fine_tuned ON models(is_fine_tuned);
CREATE TABLE _litestream_seq (
    id INTEGER PRIMARY KEY,
    seq INTEGER
);
CREATE TABLE _litestream_lock (
    id INTEGER
);
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    vector_store_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    meta JSON,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (vector_store_id) REFERENCES vector_stores (id) ON DELETE CASCADE
);
CREATE VIRTUAL TABLE vec0 USING vec0(
                document_id TEXT PRIMARY KEY,
                embedding float[384]
            );
CREATE TABLE IF NOT EXISTS "vec0_info" (key text primary key, value any);
CREATE TABLE IF NOT EXISTS "vec0_chunks"(chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,size INTEGER NOT NULL,validity BLOB NOT NULL,rowids BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS "vec0_rowids"(rowid INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE NOT NULL,chunk_id INTEGER,chunk_offset INTEGER);
CREATE TABLE IF NOT EXISTS "vec0_vector_chunks00"(rowid PRIMARY KEY,vectors BLOB NOT NULL);
CREATE TABLE vector_store_files (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL DEFAULT 'vector_store.file',
    vector_store_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'in_progress',
    created_at INTEGER NOT NULL,
    last_error JSON,
    FOREIGN KEY (vector_store_id) REFERENCES vector_stores (id) ON DELETE CASCADE,
    FOREIGN KEY (file_id) REFERENCES files (id) ON DELETE CASCADE
);
CREATE INDEX idx_vector_store_files_store_id ON vector_store_files(vector_store_id);
CREATE INDEX idx_vector_store_files_file_id ON vector_store_files(file_id);
CREATE UNIQUE INDEX idx_vector_store_files_unique ON vector_store_files(vector_store_id, file_id);
-- Dbmate schema migrations
INSERT INTO "schema_migrations" (version) VALUES
  ('20250810000001'),
  ('20250810000002'),
  ('20250810000003'),
  ('20250810000004'),
  ('20250810000005'),
  ('20250810000006'),
  ('20250810000007'),
  ('20250810000008'),
  ('20250810000009'),
  ('20250812000001'),
  ('20250812000003'),
  ('20250815000003'),
  ('20250816000001');
