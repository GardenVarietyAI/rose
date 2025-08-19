use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
}
