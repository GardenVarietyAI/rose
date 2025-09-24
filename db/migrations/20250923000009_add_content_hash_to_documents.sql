-- migrate:up
ALTER TABLE documents ADD COLUMN content_hash TEXT;

-- Create index on content_hash for faster lookups
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);

-- migrate:down
DROP INDEX IF EXISTS idx_documents_content_hash;
ALTER TABLE documents DROP COLUMN content_hash;
