use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use crate::chat_templates::ChatTemplate;
use crate::device::DeviceConfig;
use crate::models::{load_reranker, Reranker};

#[pyclass]
pub struct RerankerModel {
    model: Arc<Mutex<Box<dyn Reranker>>>,
    tokenizer: Arc<Tokenizer>,
}

#[pymethods]
impl RerankerModel {
    #[new]
    #[pyo3(signature = (model_path, tokenizer_path, device=None))]
    fn py_new(model_path: String, tokenizer_path: String, device: Option<&str>) -> PyResult<Self> {
        let device_config = DeviceConfig::from_string(device)?;

        let model = load_reranker(&model_path, &device_config.device).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load reranker: {}", e))
        })?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load tokenizer: {}", e))
        })?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
        })
    }

    fn score<'py>(
        &self,
        py: Python<'py>,
        query: String,
        document: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();

        future_into_py(py, async move {
            let full_input = ChatTemplate::format_reranker_prompt(&query, &document);

            let encoding = tokenizer.encode(full_input.as_str(), false).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Tokenization error: {}", e))
            })?;

            let tokens = encoding.get_ids();

            // Pass the full formatted tokens and score with model
            let mut model_guard = model.lock().await;
            let score = model_guard
                .score(tokens, &[]) // Pass empty array for doc_tokens since we have everything in query
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Scoring error: {}", e))
                })?;

            Ok(score)
        })
    }

    fn score_batch<'py>(
        &self,
        py: Python<'py>,
        queries: Vec<String>,
        documents: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if queries.len() != documents.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Queries and documents must have the same length",
            ));
        }

        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();

        future_into_py(py, async move {
            let mut scores = Vec::with_capacity(queries.len());
            let mut model_guard = model.lock().await;

            for (query, document) in queries.iter().zip(documents.iter()) {
                let full_input = ChatTemplate::format_reranker_prompt(query, document);

                let encoding = tokenizer.encode(full_input.as_str(), false).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Tokenization error: {}", e))
                })?;

                let score = model_guard.score(encoding.get_ids(), &[]).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Scoring error: {}", e))
                })?;

                scores.push(score);
            }

            Ok(scores)
        })
    }
}
