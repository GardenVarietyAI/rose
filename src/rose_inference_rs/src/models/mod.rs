pub mod qwen2;
pub mod qwen3;
pub mod qwen3_gguf;

use anyhow::Result;
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

pub trait CausalLM: Send {
    fn forward(&mut self, input: &Tensor, past_length: usize) -> Result<Tensor>;
    fn eos_token_id(&self) -> u32;
    fn im_end_token_id(&self, tokenizer: &Tokenizer) -> u32;
}

#[derive(Debug)]
pub enum ModelKind {
    Qwen2,
    Qwen3,
    Qwen3Gguf,
}

impl ModelKind {
    pub fn from_string(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "qwen2" => Ok(Self::Qwen2),
            "qwen3" => Ok(Self::Qwen3),
            "qwen3_gguf" => Ok(Self::Qwen3Gguf),
            _ => Err(anyhow::anyhow!("Unsupported model kind: {}", s)),
        }
    }
}

pub fn load_causal_lm(
    model_kind: ModelKind,
    model_path: &str,
    device: &Device,
) -> Result<Box<dyn CausalLM>> {
    match model_kind {
        ModelKind::Qwen2 => Ok(Box::new(qwen2::Qwen2CausalLM::load(model_path, device)?)),
        ModelKind::Qwen3 => Ok(Box::new(qwen3::Qwen3UnquantizedCausalLM::load(model_path, device)?)),
        ModelKind::Qwen3Gguf => Ok(Box::new(qwen3_gguf::Qwen3CausalLM::load(model_path, device)?)),
    }
}
