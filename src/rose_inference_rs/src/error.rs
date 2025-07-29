use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
}
