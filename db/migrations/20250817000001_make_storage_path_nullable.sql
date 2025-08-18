-- migrate:up
ALTER TABLE files ADD COLUMN storage_path_new TEXT;
UPDATE files SET storage_path_new = NULL;
ALTER TABLE files DROP COLUMN storage_path;
ALTER TABLE files RENAME COLUMN storage_path_new TO storage_path;

-- migrate:down
ALTER TABLE files ADD COLUMN storage_path_new TEXT NOT NULL DEFAULT 'BLOB';
UPDATE files SET storage_path_new = COALESCE(storage_path, 'BLOB');
ALTER TABLE files DROP COLUMN storage_path;
ALTER TABLE files RENAME COLUMN storage_path_new TO storage_path;
