-- migrate:up
-- Add BLOB storage support to files table
ALTER TABLE files ADD COLUMN content BLOB;

-- migrate:down
-- Remove BLOB column (note: this will lose BLOB data)
ALTER TABLE files DROP COLUMN content;
