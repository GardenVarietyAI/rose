use std::sync::{Arc, OnceLock};
use tokio::sync::Mutex;
use pyo3::prelude::*;

use crate::device::DeviceConfig;
use crate::models::{CausalLM, ModelKind, load_causal_lm};
use crate::types::InferenceResponse;
use tokio::sync::mpsc;

pub struct ModelEntry {
    pub key: (String, String), // (path, device_kind)
    pub model: Arc<Mutex<Box<dyn CausalLM>>>,
}

static CACHED_MODEL: OnceLock<Arc<Mutex<Option<ModelEntry>>>> = OnceLock::new();

pub struct ModelCache;

impl ModelCache {
    pub fn init() {
        CACHED_MODEL.get_or_init(|| Arc::new(Mutex::new(None)));
    }

    pub async fn get_or_load_model(
        model_kind: ModelKind,
        model_path: &str,
        device_config: &DeviceConfig,
        tx: &mpsc::Sender<InferenceResponse>,
    ) -> PyResult<Option<Arc<Mutex<Box<dyn CausalLM>>>>> {
        let cache = CACHED_MODEL.get().unwrap();
        let mut model_cache = cache.lock().await;
        let device_str = DeviceConfig::device_kind(&device_config.device);
        let cache_key = (model_path.to_string(), device_str);

        // Check if we have a cached model for this path+device
        if let Some(ref entry) = *model_cache {
            if entry.key == cache_key {
                return Ok(Some(entry.model.clone()));
            }
        }

        // Load new model
        match load_causal_lm(model_kind, model_path, &device_config.device) {
            Ok(m) => {
                let shared_model = Arc::new(Mutex::new(m));
                *model_cache = Some(ModelEntry {
                    key: cache_key,
                    model: shared_model.clone(),
                });
                Ok(Some(shared_model))
            }
            Err(e) => {
                let error_msg = format!(
                    "Failed to load model '{}' on device '{}': {}",
                    model_path,
                    DeviceConfig::device_kind(&device_config.device),
                    e
                );
                tracing::error!("{}", error_msg);
                let _ = tx.send(InferenceResponse::Error { error: error_msg }).await;
                Ok(None)
            }
        }
    }

    pub async fn flush_model() -> PyResult<()> {
        if let Some(cache) = CACHED_MODEL.get() {
            let mut model_cache = cache.lock().await;
            *model_cache = None;
        }
        Ok(())
    }
}
