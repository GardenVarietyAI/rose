-- migrate:up
CREATE INDEX IF NOT EXISTS idx_vsf_lookup
ON vector_store_files(vector_store_id, file_id);
CREATE INDEX IF NOT EXISTS idx_vsf_status
ON vector_store_files(vector_store_id, status);

CREATE INDEX IF NOT EXISTS idx_doc_vs_file
ON documents(vector_store_id, file_id);
CREATE INDEX IF NOT EXISTS idx_doc_vs_created
ON documents(vector_store_id, created_at);

-- migrate:down
DROP INDEX IF EXISTS idx_vsf_lookup;
DROP INDEX IF EXISTS idx_vsf_status;
DROP INDEX IF EXISTS idx_doc_vs_file;
DROP INDEX IF EXISTS idx_doc_vs_created;
