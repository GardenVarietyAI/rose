use candle_core::Device;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_async_runtimes::tokio as pyo3_tokio;
use pyo3_async_runtimes::tokio::future_into_py;
use std::sync::{Arc, OnceLock};
use tokio::sync::Mutex;

mod chat_templates;
mod error;
mod generate;
mod logprobs;
mod models;
mod types;

use crate::models::{CausalLM, ModelKind};
use crate::types::{InferenceRequest, InferenceResponse, Message};

struct ModelEntry {
    key: (String, String), // (path, device_kind)
    model: Arc<Mutex<Box<dyn CausalLM>>>,
}

static CACHED_MODEL: OnceLock<Arc<Mutex<Option<ModelEntry>>>> = OnceLock::new();

macro_rules! send_error {
    ($cb:expr, $($arg:tt)*) => {
        {
            let error_msg = format!($($arg)*);
            tracing::error!("{}", error_msg);
            Python::with_gil(|py| {
                let d = PyDict::new(py);
                let _ = d.set_item("type", "Error");
                let _ = d.set_item("error", &error_msg);
                let _ = $cb.bind(py).call1((d,));
            });
        }
    };
}

#[pymodule]
fn _inference(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Build a Tokio runtime and register it
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        tracing_subscriber::fmt::init();

        let rt = ::tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        // Hand the runtime to pyo3-async-runtimes
        let _ = pyo3_tokio::init_with_runtime(Box::leak(Box::new(rt)));
    });

    m.add_class::<InferenceServer>()?;
    Ok(())
}

fn device_kind(device: &Device) -> String {
    match device {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(id) => format!("cuda:{:?}", id),
        Device::Metal(id) => format!("metal:{:?}", id),
    }
}

fn format_messages(messages: &[Message], template: Option<&str>) -> String {
    let chat_template =
        crate::chat_templates::ChatTemplate::from_string(template.unwrap_or("qwen3"));
    chat_template.format_messages(messages)
}

#[pyclass]
pub struct InferenceServer {
    device: Device,
}
#[pymethods]
impl InferenceServer {
    #[new]
    #[pyo3(signature = (device=None))]
    fn py_new(device: Option<&str>) -> PyResult<Self> {
        let resolved = match device.unwrap_or("auto") {
            "cpu" => Device::Cpu,
            #[cfg(feature = "cuda")]
            "cuda" => candle_core::Device::new_cuda(0)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
            #[cfg(feature = "metal")]
            "metal" => candle_core::Device::new_metal(0)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
            #[cfg(not(feature = "metal"))]
            "metal" => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Metal support not compiled",
                ))
            }
            _ => {
                #[cfg(feature = "metal")]
                {
                    if let Ok(d) = candle_core::Device::new_metal(0) {
                        d
                    } else {
                        Device::Cpu
                    }
                }
                #[cfg(not(feature = "metal"))]
                Device::Cpu
            }
        };

        CACHED_MODEL.get_or_init(|| Arc::new(Mutex::new(None)));

        Ok(Self { device: resolved })
    }

    fn flush_model<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let cache_clone = CACHED_MODEL.get().unwrap().clone();

        future_into_py(py, async move {
            let mut model_cache = cache_clone.lock().await;
            *model_cache = None;
            Ok(Python::with_gil(|py| py.None()))
        })
    }

    #[pyo3(signature = (request, on_event))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        request: PyObject,
        on_event: PyObject,
    ) -> PyResult<Bound<'py, PyAny>> {
        let req: InferenceRequest = pythonize::depythonize(&request.bind(py))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad request: {e}")))?;
        let device = self.device.clone();
        let cb = on_event.clone_ref(py);

        future_into_py(py, async move {
            tracing::info!("Starting stream inference");
            let have_prompt = req.prompt.as_ref().map_or(false, |p| !p.is_empty());
            let have_msgs = req.messages.as_ref().map_or(false, |m| !m.is_empty());
            if (have_prompt && have_msgs) || (!have_prompt && !have_msgs) {
                send_error!(cb, "Provide either prompt OR messages (exclusively)");
                return Ok(Python::with_gil(|py| py.None()));
            }

            let prompt_str = if let Some(p) = &req.prompt {
                p.clone()
            } else {
                format_messages(
                    req.messages.as_ref().unwrap(),
                    req.generation_kwargs.chat_template.as_deref(),
                )
            };

            let model_kind = match ModelKind::from_string(&req.generation_kwargs.model_kind) {
                Ok(kind) => kind,
                Err(e) => {
                    send_error!(
                        cb,
                        "Invalid model kind '{}': {}",
                        req.generation_kwargs.model_kind,
                        e
                    );
                    return Ok(Python::with_gil(|py| py.None()));
                }
            };

            let tokenizer = {
                let tokenizer_path = std::path::Path::new(&req.generation_kwargs.tokenizer_path);
                match tokenizers::Tokenizer::from_file(&tokenizer_path) {
                    Ok(t) => t,
                    Err(e) => {
                        send_error!(
                            cb,
                            "Failed to load tokenizer from {}: {}",
                            tokenizer_path.display(),
                            e
                        );
                        return Ok(Python::with_gil(|py| py.None()));
                    }
                }
            };

            let model = {
                let cache = CACHED_MODEL.get().unwrap();
                let mut model_cache = cache.lock().await;
                let device_str = device_kind(&device);
                let cache_key = (req.generation_kwargs.model_path.clone(), device_str);

                // Check if we have a cached model for this path+device
                if let Some(ref entry) = *model_cache {
                    if entry.key == cache_key {
                        entry.model.clone()
                    } else {
                        // Different model path or device, load new one
                        match models::load_causal_lm(
                            model_kind,
                            &req.generation_kwargs.model_path,
                            &device,
                        ) {
                            Ok(m) => {
                                let shared_model = Arc::new(Mutex::new(m));
                                *model_cache = Some(ModelEntry {
                                    key: cache_key,
                                    model: shared_model.clone(),
                                });
                                shared_model
                            }
                            Err(e) => {
                                send_error!(
                                    cb,
                                    "Failed to load model '{}' on device '{}': {}",
                                    req.generation_kwargs.model_path,
                                    device_kind(&device),
                                    e
                                );
                                return Ok(Python::with_gil(|py| py.None()));
                            }
                        }
                    }
                } else {
                    // No cached model, load new one
                    match models::load_causal_lm(
                        model_kind,
                        &req.generation_kwargs.model_path,
                        &device,
                    ) {
                        Ok(m) => {
                            let shared_model = Arc::new(Mutex::new(m));
                            *model_cache = Some(ModelEntry {
                                key: cache_key,
                                model: shared_model.clone(),
                            });
                            shared_model
                        }
                        Err(e) => {
                            send_error!(
                                cb,
                                "Failed to load model '{}' on device '{}': {}",
                                req.generation_kwargs.model_path,
                                device_kind(&device),
                                e
                            );
                            return Ok(Python::with_gil(|py| py.None()));
                        }
                    }
                }
            };

            let max_in = req.generation_kwargs.max_input_tokens.unwrap_or(8192);
            let max_out = req.generation_kwargs.max_output_tokens.unwrap_or(1024);
            let temp = req.generation_kwargs.temperature.unwrap_or(0.7) as f32;
            let top_p = req.generation_kwargs.top_p.map(|v| v as f32);
            let seed = req.generation_kwargs.seed.unwrap_or(42);
            let rp = req.generation_kwargs.repeat_penalty.unwrap_or(1.1);
            let rlast = req.generation_kwargs.repeat_last_n.unwrap_or(64);

            let (tx, mut rx) = ::tokio::sync::mpsc::channel::<InferenceResponse>(1);

            let gen = ::tokio::spawn({
                let model = model; // move it into the task
                let tokenizer = tokenizer.clone();
                let device = device.clone();
                let prompt = prompt_str.clone();
                async move {
                    let mut model_guard = model.lock().await;
                    let _ = generate::stream(
                        model_guard.as_mut(),
                        &tokenizer,
                        device,
                        &prompt,
                        max_in,
                        max_out,
                        temp,
                        top_p,
                        req.generation_kwargs.stop.as_deref(),
                        tx,
                        seed,
                        rp,
                        rlast,
                        req.generation_kwargs.logprobs,
                        req.generation_kwargs.top_logprobs,
                    )
                    .await;
                }
            });

            // Add a 5 minute timeout for generation task
            let timeout_duration = std::time::Duration::from_secs(300);
            let gen_with_timeout = tokio::time::timeout(timeout_duration, gen);

            while let Some(ev) = rx.recv().await {
                let is_terminal = matches!(
                    ev,
                    InferenceResponse::Complete { .. } | InferenceResponse::Error { .. }
                );

                // Call Python callback immediately without spawn_blocking to avoid Send issues
                Python::with_gil(|py| {
                    let d = PyDict::new(py);
                    match &ev {
                        InferenceResponse::InputTokensCounted { input_tokens } => {
                            d.set_item("type", "InputTokensCounted").ok();
                            d.set_item("input_tokens", *input_tokens).ok();
                        }
                        InferenceResponse::Token {
                            token,
                            position,
                            logprob,
                            top_logprobs,
                        } => {
                            d.set_item("type", "Token").ok();
                            d.set_item("token", token.as_str()).ok();
                            d.set_item("position", *position).ok();
                            if let Some(lp) = *logprob {
                                d.set_item("logprob", lp).ok();
                            }
                            if let Some(tlps) = top_logprobs {
                                if let Ok(py_tlps) = pythonize::pythonize(py, tlps) {
                                    d.set_item("top_logprobs", py_tlps).ok();
                                }
                            }
                        }
                        InferenceResponse::Complete {
                            input_tokens,
                            output_tokens,
                            total_tokens,
                            finish_reason,
                        } => {
                            d.set_item("type", "Complete").ok();
                            d.set_item("input_tokens", *input_tokens).ok();
                            d.set_item("output_tokens", *output_tokens).ok();
                            d.set_item("total_tokens", *total_tokens).ok();
                            let reason = match finish_reason {
                                crate::types::FinishReason::Stop => "stop",
                                crate::types::FinishReason::Length => "length",
                                crate::types::FinishReason::ToolCalls => "tool_calls",
                                crate::types::FinishReason::ContentFilter => "content_filter",
                                crate::types::FinishReason::FunctionCall => "function_call",
                            };
                            d.set_item("finish_reason", reason).ok();
                        }
                        InferenceResponse::Error { error } => {
                            d.set_item("type", "Error").ok();
                            d.set_item("error", error.as_str()).ok();
                        }
                    }
                    if let Err(err) = cb.bind(py).call1((d,)) {
                        err.print(py);
                    }
                });

                if is_terminal {
                    break;
                }
            }
            match gen_with_timeout.await {
                Ok(Ok(_)) => {
                    tracing::info!("Inference generation completed successfully");
                }
                Ok(Err(e)) => tracing::error!("Generation task failed: {:?}", e),
                Err(_) => {
                    tracing::error!(
                        "Generation timed out after {} seconds",
                        timeout_duration.as_secs()
                    );
                    // Send timeout error through callback
                    Python::with_gil(|py| {
                        let d = PyDict::new(py);
                        let _ = d.set_item("type", "Error");
                        let _ = d.set_item("error", "Generation timed out");
                        let _ = cb.bind(py).call1((d,));
                    });
                }
            }

            Ok(Python::with_gil(|py| py.None()))
        })
    }
}
