-- migrate:up
-- Create Litestream internal tables for replication tracking
-- These tables are used by Litestream to track replication state and sequence numbers

CREATE TABLE IF NOT EXISTS _litestream_seq (
    id INTEGER PRIMARY KEY,
    seq INTEGER
);

CREATE TABLE IF NOT EXISTS _litestream_lock (
    id INTEGER
);

-- migrate:down
-- Remove Litestream tables (this will break active replication)
DROP TABLE IF EXISTS _litestream_lock;
DROP TABLE IF EXISTS _litestream_seq;