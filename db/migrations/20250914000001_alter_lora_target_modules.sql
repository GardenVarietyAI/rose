-- migrate:up
ALTER TABLE models RENAME COLUMN lora_target_modules TO lora_target_modules_text;
ALTER TABLE models ADD COLUMN lora_target_modules JSON;
UPDATE models SET lora_target_modules = lora_target_modules_text;
ALTER TABLE models DROP COLUMN lora_target_modules_text;

-- migrate:down
ALTER TABLE models RENAME COLUMN lora_target_modules TO lora_target_modules_json;
ALTER TABLE models ADD COLUMN lora_target_modules TEXT;
UPDATE models SET lora_target_modules = lora_target_modules_json;
ALTER TABLE models DROP COLUMN lora_target_modules_json;
