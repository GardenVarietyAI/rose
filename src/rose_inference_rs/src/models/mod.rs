pub mod qwen2;

use anyhow::Result;
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

pub trait CausalLM: Send {
    fn forward(&mut self, input: &Tensor, past_length: usize) -> Result<Tensor>;
    fn sample_logits(&self, logits: &Tensor) -> Result<u32>;
    fn eos_token_id(&self) -> u32;
    fn im_end_token_id(&self, tokenizer: &Tokenizer) -> u32;
}

pub enum ModelKind {
    Qwen2,
}

pub fn load_causal_lm(
    model_kind: ModelKind,
    model_path: &str,
    device: &Device,
) -> Result<Box<dyn CausalLM>> {
    match model_kind {
        ModelKind::Qwen2 => Ok(Box::new(qwen2::Qwen2CausalLM::load(model_path, device)?)),
    }
}
