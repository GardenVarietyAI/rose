-- Seed default models for ROSE Server
-- This file contains base model configurations that can be loaded into the database

-- Existing models
INSERT INTO "models" ("id", "name", "model_name", "model_type", "path", "is_fine_tuned", "temperature", "top_p", "memory_gb", "timeout", "lora_target_modules", "owned_by", "root", "parent", "permissions", "created_at") VALUES
('phi-1.5', 'phi-1.5', 'microsoft/phi-1_5', 'huggingface', NULL, '0', '0.7', '0.95', '2.5', NULL, '["q_proj", "k_proj", "v_proj", "dense"]', 'microsoft', 'phi-1.5', NULL, '[]', '1750698675'),
('phi-2', 'phi-2', 'microsoft/phi-2', 'huggingface', NULL, '0', '0.5', '0.9', '5.0', NULL, '["q_proj", "k_proj", "v_proj", "dense"]', 'microsoft', 'phi-2', NULL, '[]', '1750698675'),
('qwen-coder', 'qwen-coder', 'Qwen/Qwen2.5-Coder-1.5B-Instruct', 'huggingface', NULL, '0', '0.2', '0.9', '3.0', '90', '["q_proj", "k_proj", "v_proj", "o_proj"]', 'alibaba', 'qwen-coder', NULL, '[]', '1750698675'),
('qwen2.5-0.5b', 'qwen2.5-0.5b', 'Qwen/Qwen2.5-0.5B-Instruct', 'huggingface', NULL, '0', '0.3', '0.9', '1.5', '60', '["q_proj", "k_proj", "v_proj", "o_proj"]', 'alibaba', 'qwen2.5-0.5b', NULL, '[]', '1750698675'),
('tinyllama', 'tinyllama', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'huggingface', NULL, '0', '0.4', '0.9', '2.0', NULL, '["q_proj", "k_proj", "v_proj", "o_proj"]', 'organization-owner', 'tinyllama', NULL, '[]', '1750698675'),

-- Popular models to show reach
('llama-3.2-1b', 'Llama 3.2 1B', 'meta-llama/Llama-3.2-1B-Instruct', 'huggingface', NULL, '0', '0.7', '0.9', '2.5', '120', '["q_proj", "k_proj", "v_proj", "o_proj"]', 'meta', 'llama-3.2-1b', NULL, '[]', '1750698675'),
('llama-3.2-3b', 'Llama 3.2 3B', 'meta-llama/Llama-3.2-3B-Instruct', 'huggingface', NULL, '0', '0.7', '0.9', '6.0', '180', '["q_proj", "k_proj", "v_proj", "o_proj"]', 'meta', 'llama-3.2-3b', NULL, '[]', '1750698675'),
('llama-3.1-8b', 'Llama 3.1 8B', 'meta-llama/Llama-3.1-8B-Instruct', 'huggingface', NULL, '0', '0.7', '0.9', '16.0', '300', '["q_proj", "k_proj", "v_proj", "o_proj"]', 'meta', 'llama-3.1-8b', NULL, '[]', '1750698675'),
('mistral-7b', 'Mistral 7B', 'mistralai/Mistral-7B-Instruct-v0.3', 'huggingface', NULL, '0', '0.7', '0.9', '14.0', '300', '["q_proj", "k_proj", "v_proj", "o_proj"]', 'mistral-ai', 'mistral-7b', NULL, '[]', '1750698675'),
('mixtral-8x7b', 'Mixtral 8x7B', 'mistralai/Mixtral-8x7B-Instruct-v0.1', 'huggingface', NULL, '0', '0.7', '0.9', '90.0', '600', '["q_proj", "k_proj", "v_proj", "o_proj"]', 'mistral-ai', 'mixtral-8x7b', NULL, '[]', '1750698675'),
('deepseek-1.3b', 'DeepSeek 1.3B', 'deepseek-ai/deepseek-coder-1.3b-instruct', 'huggingface', NULL, '0', '0.3', '0.95', '3.0', '120', '["q_proj", "k_proj", "v_proj", "o_proj"]', 'deepseek', 'deepseek-1.3b', NULL, '[]', '1750698675'),
('deepseek-7b', 'DeepSeek 7B', 'deepseek-ai/deepseek-coder-7b-instruct-v1.5', 'huggingface', NULL, '0', '0.3', '0.95', '14.0', '300', '["q_proj", "k_proj", "v_proj", "o_proj"]', 'deepseek', 'deepseek-7b', NULL, '[]', '1750698675'),
('qwen2.5-1.5b', 'Qwen2.5 1.5B', 'Qwen/Qwen2.5-1.5B-Instruct', 'huggingface', NULL, '0', '0.7', '0.9', '3.0', '120', '["q_proj", "k_proj", "v_proj", "o_proj"]', 'alibaba', 'qwen2.5-1.5b', NULL, '[]', '1750698675'),
('qwen2.5-3b', 'Qwen2.5 3B', 'Qwen/Qwen2.5-3B-Instruct', 'huggingface', NULL, '0', '0.7', '0.9', '6.0', '180', '["q_proj", "k_proj", "v_proj", "o_proj"]', 'alibaba', 'qwen2.5-3b', NULL, '[]', '1750698675'),
('qwen2.5-7b', 'Qwen2.5 7B', 'Qwen/Qwen2.5-7B-Instruct', 'huggingface', NULL, '0', '0.7', '0.9', '14.0', '300', '["q_proj", "k_proj", "v_proj", "o_proj"]', 'alibaba', 'qwen2.5-7b', NULL, '[]', '1750698675'),
('gemma-2b', 'Gemma 2B', 'google/gemma-2b-it', 'huggingface', NULL, '0', '0.7', '0.9', '5.0', '180', '["q_proj", "k_proj", "v_proj", "o_proj"]', 'google', 'gemma-2b', NULL, '[]', '1750698675'),
('starcoder2-3b', 'StarCoder2 3B', 'bigcode/starcoder2-3b', 'huggingface', NULL, '0', '0.2', '0.95', '6.0', '180', '["q_proj", "k_proj", "v_proj", "o_proj"]', 'bigcode', 'starcoder2-3b', NULL, '[]', '1750698675'),
('starcoder2-7b', 'StarCoder2 7B', 'bigcode/starcoder2-7b', 'huggingface', NULL, '0', '0.2', '0.95', '14.0', '300', '["q_proj", "k_proj", "v_proj", "o_proj"]', 'bigcode', 'starcoder2-7b', NULL, '[]', '1750698675');

-- Note: Larger models (>7B) may require significant GPU memory and are included here
-- for reference but may not run on consumer hardware without quantization
