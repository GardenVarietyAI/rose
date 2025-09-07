use std::sync::{Arc, OnceLock};
use std::collections::HashMap;
use tokio::sync::Mutex;
use pyo3::prelude::*;

use crate::device::DeviceConfig;
use crate::models::{CausalLM, ModelKind, load_causal_lm};
use crate::types::InferenceResponse;
use tokio::sync::mpsc;

pub struct ModelEntry {
    pub key: (String, String), // (path, device_kind)
    pub model: Arc<Mutex<Box<dyn CausalLM>>>,
    pub conversations: HashMap<String, ConversationState>,
}

#[derive(Clone)]
pub struct ConversationState {
    pub active: bool,
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
                    conversations: HashMap::new(),
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
            if let Some(ref entry) = *model_cache {
                // Force reset state before dropping
                if let Ok(mut model) = entry.model.try_lock() {
                    model.reset_state();
                }
            }
            *model_cache = None;
        }
        Ok(())
    }

    pub async fn should_reset_kv_cache(response_chain_id: Option<&str>) -> bool {
        let Some(chain_id) = response_chain_id else {
            return true; // Always reset if no chain ID
        };

        let Some(cache) = CACHED_MODEL.get() else {
            return true;
        };

        let Ok(model_cache) = cache.try_lock() else {
            return true; // Reset if can't acquire lock
        };

        let Some(ref entry) = *model_cache else {
            return true;
        };

        // Reset if conversation is NOT active, don't reset if it IS active
        entry.conversations.get(chain_id).map_or(true, |state| !state.active)
    }

    pub async fn mark_conversation_active(response_chain_id: Option<&str>) {
        let Some(chain_id) = response_chain_id else {
            return;
        };

        let Some(cache) = CACHED_MODEL.get() else {
            return;
        };

        let Ok(mut model_cache) = cache.try_lock() else {
            return;
        };

        if let Some(ref mut entry) = *model_cache {
            entry.conversations.insert(chain_id.to_string(), ConversationState {
                active: true,
            });
        }
    }
}
