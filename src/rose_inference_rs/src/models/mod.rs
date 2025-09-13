pub mod qwen3;
pub mod qwen3_gguf;
pub mod qwen3_lora;

use anyhow::Result;
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

pub trait CausalLM: Send {
    fn forward(&mut self, input: &Tensor, past_length: usize) -> Result<Tensor>;
    fn eos_token_id(&self) -> u32;
    fn im_end_token_id(&self, tokenizer: &Tokenizer) -> u32;
    fn reset_state(&mut self);
}

#[derive(Debug)]
pub enum ModelKind {
    Qwen3,
    Qwen3Gguf,
    Qwen3Lora,
}

impl ModelKind {
    pub fn from_string(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "qwen3" => Ok(Self::Qwen3),
            "qwen3_gguf" => Ok(Self::Qwen3Gguf),
            "qwen3_lora" => Ok(Self::Qwen3Lora),
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
            model_path, lora_adapter_path, device,
        )?)),
    }
}
