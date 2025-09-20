pub mod qwen3;
pub mod qwen3_gguf;
pub mod qwen3_lora;
pub mod qwen3_reranker;

use anyhow::Result;
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

pub trait CausalLM: Send {
    fn forward(&mut self, input: &Tensor, past_length: usize) -> Result<Tensor>;
    fn eos_token_id(&self) -> u32;
    fn im_end_token_id(&self, tokenizer: &Tokenizer) -> u32;
    fn reset_state(&mut self);
}

pub trait Reranker: Send {
    fn score(&mut self, query_tokens: &[u32], doc_tokens: &[u32]) -> Result<f32>;
    fn forward(&mut self, input: &Tensor) -> Result<Tensor>;
}

#[derive(Debug)]
pub enum ModelKind {
    Qwen3,
    Qwen3Gguf,
    Qwen3Lora,
    Qwen3RerankerGguf,
}

impl ModelKind {
    pub fn from_string(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "qwen3" => Ok(Self::Qwen3),
            "qwen3_gguf" => Ok(Self::Qwen3Gguf),
            "qwen3_lora" => Ok(Self::Qwen3Lora),
            "qwen3_reranker" | "qwen3_reranker_gguf" => Ok(Self::Qwen3RerankerGguf),
            _ => Err(anyhow::anyhow!("Unsupported model kind: {}", s)),
        }
    }
}

#[allow(dead_code)]
pub fn load_causal_lm(
    model_kind: ModelKind,
    model_path: &str,
    device: &Device,
) -> Result<Box<dyn CausalLM>> {
    load_causal_lm_with_lora(model_kind, model_path, None, device)
}

pub fn load_causal_lm_with_lora(
    model_kind: ModelKind,
    model_path: &str,
    lora_adapter_path: Option<&str>,
    device: &Device,
) -> Result<Box<dyn CausalLM>> {
    match model_kind {
        ModelKind::Qwen3 => Ok(Box::new(qwen3::Qwen3UnquantizedCausalLM::load(
            model_path, device,
        )?)),
        ModelKind::Qwen3Gguf => Ok(Box::new(qwen3_gguf::Qwen3CausalLM::load(
            model_path, device,
        )?)),
        ModelKind::Qwen3Lora => Ok(Box::new(qwen3_lora::Qwen3LoraModel::load(
            model_path,
            lora_adapter_path,
            device,
        )?)),
        ModelKind::Qwen3RerankerGguf => {
            Err(anyhow::anyhow!("Use load_reranker for reranker models"))
        }
    }
}

pub fn load_reranker(model_path: &str, device: &Device) -> Result<Box<dyn Reranker>> {
    Ok(Box::new(qwen3_reranker::Qwen3Reranker::load(
        model_path, device,
    )?))
}
