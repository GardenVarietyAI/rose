-- migrate:up
ALTER TABLE documents ADD COLUMN vector_store_file_id TEXT REFERENCES vector_store_files(id);
CREATE INDEX idx_documents_vector_store_file_id ON documents(vector_store_file_id);

ALTER TABLE vector_store_files ADD COLUMN attributes JSON;

-- migrate:down
ALTER TABLE documents DROP COLUMN vector_store_file_id;
DROP INDEX IF EXISTS idx_documents_vector_store_file_id;
ALTER TABLE vector_store_files DROP COLUMN attributes;
