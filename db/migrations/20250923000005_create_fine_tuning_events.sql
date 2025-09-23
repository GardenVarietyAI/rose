-- migrate:up
CREATE TABLE IF NOT EXISTS fine_tuning_events (
    id TEXT PRIMARY KEY,
    object TEXT,
    job_id TEXT NOT NULL REFERENCES fine_tuning_jobs(id),
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    data JSON,
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fine_tuning_events_job_id ON fine_tuning_events(job_id);
CREATE INDEX IF NOT EXISTS idx_fine_tuning_events_created_at ON fine_tuning_events(created_at);

-- migrate:down
DROP INDEX IF EXISTS idx_fine_tuning_events_created_at;
DROP INDEX IF EXISTS idx_fine_tuning_events_job_id;
DROP TABLE fine_tuning_events;
