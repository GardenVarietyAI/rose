-- migrate:up
ALTER TABLE models ADD COLUMN kind TEXT;

-- migrate:down
ALTER TABLE models DROP COLUMN kind;
