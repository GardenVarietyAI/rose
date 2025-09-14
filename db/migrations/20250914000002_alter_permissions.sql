-- migrate:up
ALTER TABLE models RENAME COLUMN permissions TO permissions_text;
ALTER TABLE models ADD COLUMN permissions JSON;
UPDATE models SET permissions = permissions_text;
ALTER TABLE models DROP COLUMN permissions_text;

-- migrate:down
ALTER TABLE models RENAME COLUMN permissions TO permissions_json;
ALTER TABLE models ADD COLUMN permissions TEXT;
UPDATE models SET permissions = permissions_json;
ALTER TABLE models DROP COLUMN permissions_json;
