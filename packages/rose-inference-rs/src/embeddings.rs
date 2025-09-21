use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use crate::device::DeviceConfig;
use crate::models::{load_embeddings_with_dims, Embeddings};

#[pyclass]
pub struct EmbeddingModel {
    model: Arc<Mutex<Box<dyn Embeddings>>>,
    tokenizer: Arc<Tokenizer>,
}

#[pymethods]
impl EmbeddingModel {
    #[new]
    #[pyo3(signature = (model_path, tokenizer_path, device=None, output_dims=None))]
    fn py_new(model_path: String, tokenizer_path: String, device: Option<&str>, output_dims: Option<usize>) -> PyResult<Self> {
        let device_config = DeviceConfig::from_string(device)?;

        let model = load_embeddings_with_dims(&model_path, &device_config.device, output_dims).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load embeddings: {}", e))
        })?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load tokenizer: {}", e))
        })?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
        })
    }

    fn encode<'py>(&self, py: Python<'py>, text: String) -> PyResult<Bound<'py, PyAny>> {
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();

        future_into_py(py, async move {
            let encoding = tokenizer.encode(text.as_str(), false).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Tokenization error: {}", e))
            })?;

            let tokens = encoding.get_ids();

            let mut model_guard = model.lock().await;
            let embedding = model_guard.encode(tokens).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Encoding error: {}", e))
            })?;

            Ok(embedding)
        })
    }

    fn encode_batch<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();

        future_into_py(py, async move {
            let mut token_batches = Vec::with_capacity(texts.len());
            let mut total_tokens = 0usize;

            for text in texts {
                let encoding = tokenizer.encode(text.as_str(), false).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Tokenization error: {}", e))
                })?;

                let tokens = encoding.get_ids();
                total_tokens += tokens.len();
                token_batches.push(tokens.to_vec());
            }

            let mut model_guard = model.lock().await;
            let embeddings = model_guard.encode_batch(token_batches).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Batch encoding error: {}", e))
            })?;

            Ok((embeddings, total_tokens))
        })
    }
}
