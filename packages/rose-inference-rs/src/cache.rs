use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use tokio::sync::Mutex;

use crate::device::DeviceConfig;
use crate::models::{load_causal_lm_with_lora, CausalLM, ModelKind};
use crate::types::InferenceResponse;
use tokio::sync::mpsc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CacheKey {
    model_kind: String,
    model_path: String,
    device_kind: String,
    lora_adapter_path: Option<String>,
}

impl CacheKey {
    fn new(
        model_kind: &ModelKind,
        model_path: &str,
        device_kind: &str,
        lora_adapter_path: Option<&str>,
    ) -> Self {
        let model_kind_str = match model_kind {
            ModelKind::Qwen3 => "qwen3",
            ModelKind::Qwen3Gguf => "qwen3_gguf",
            ModelKind::Qwen3Lora => "qwen3_lora",
        };
        Self {
            model_kind: model_kind_str.to_string(),
            model_path: model_path.to_string(),
            device_kind: device_kind.to_string(),
            lora_adapter_path: lora_adapter_path.map(|s| s.to_string()),
        }
    }
}

pub struct ModelEntry {
    pub key: CacheKey,
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
        lora_adapter_path: Option<&str>,
        device_config: &DeviceConfig,
        tx: &mpsc::Sender<InferenceResponse>,
    ) -> PyResult<Option<Arc<Mutex<Box<dyn CausalLM>>>>> {
        let cache = CACHED_MODEL.get().unwrap();
        let mut model_cache = cache.lock().await;
        let device_str = DeviceConfig::device_kind(&device_config.device);
        let cache_key = CacheKey::new(&model_kind, model_path, &device_str, lora_adapter_path);

        // Check if we have a cached model for this path+device
        if let Some(ref entry) = *model_cache {
            if entry.key == cache_key {
                tracing::info!(
                    "Reusing cached model: kind={} path={} device={} lora={}",
                    entry.key.model_kind,
                    entry.key.model_path,
                    entry.key.device_kind,
                    entry.key.lora_adapter_path.as_deref().unwrap_or("<none>")
                );
                return Ok(Some(entry.model.clone()));
            } else {
                tracing::info!(
                    "Cache miss or different adapter, reloading. Old: kind={} path={} device={} lora={} | New: kind={} path={} device={} lora={}",
                    entry.key.model_kind,
                    entry.key.model_path,
                    entry.key.device_kind,
                    entry.key.lora_adapter_path.as_deref().unwrap_or("<none>"),
                    cache_key.model_kind,
                    cache_key.model_path,
                    cache_key.device_kind,
                    cache_key.lora_adapter_path.as_deref().unwrap_or("<none>")
                );
            }
        }

        // Load new model with quantization-aware device config
        let is_quantized = matches!(model_kind, ModelKind::Qwen3Gguf);
        let optimized_device_config = if is_quantized {
            crate::device::DeviceConfig::detect_dtypes_for_quantized(&device_config.device, true)
        } else {
            device_config.clone()
        };

        match load_causal_lm_with_lora(
            model_kind,
            model_path,
            lora_adapter_path,
            &optimized_device_config.device,
        ) {
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
        entry
            .conversations
            .get(chain_id)
            .map_or(true, |state| !state.active)
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
            entry
                .conversations
                .insert(chain_id.to_string(), ConversationState { active: true });
        }
    }
}
