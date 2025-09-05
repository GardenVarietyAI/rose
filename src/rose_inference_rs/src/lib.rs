use pyo3::prelude::*;
use pyo3_async_runtimes::tokio as pyo3_tokio;
use pyo3_async_runtimes::tokio::future_into_py;

mod cache;
mod chat_templates;
mod device;
mod error;
mod generate;
mod logprobs;
mod models;
mod types;

use crate::cache::ModelCache;
use crate::device::DeviceConfig;
use crate::models::ModelKind;
use crate::types::{InferenceRequest, InferenceResponse, Message, TokenEvent, CompleteEvent, InputTokensCountedEvent, ErrorEvent, TopLogProb};



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
    m.add_class::<TokenEvent>()?;
    m.add_class::<CompleteEvent>()?;
    m.add_class::<InputTokensCountedEvent>()?;
    m.add_class::<ErrorEvent>()?;
    m.add_class::<TopLogProb>()?;
    Ok(())
}

fn format_messages(
    messages: &[Message],
    template: Option<&str>,
    enable_thinking: Option<bool>,
) -> String {
    let chat_template =
        crate::chat_templates::ChatTemplate::from_string(template.unwrap_or("qwen3"));
    chat_template.format_messages(messages, enable_thinking)
}

#[pyclass]
pub struct InferenceServer {
    device_config: DeviceConfig,
}
#[pymethods]
impl InferenceServer {
    #[new]
    #[pyo3(signature = (device=None))]
    fn py_new(device: Option<&str>) -> PyResult<Self> {
        let device_config = DeviceConfig::from_string(device)?;
        ModelCache::init();
        Ok(Self { device_config })
    }

    fn flush_model<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        future_into_py(py, async move {
            ModelCache::flush_model().await?;
            Ok(Python::attach(|py| py.None()))
        })
    }

    #[pyo3(signature = (request, on_event))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        request: Py<PyAny>,
        on_event: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let req: InferenceRequest = pythonize::depythonize(&request.bind(py))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad request: {e}")))?;
        let device_config = self.device_config.clone();
        let cb = on_event.clone_ref(py);

        future_into_py(py, async move {
            tracing::info!("Starting stream inference");

            // Create channel first
            let (tx, mut rx) = ::tokio::sync::mpsc::channel::<InferenceResponse>(1);

            let have_prompt = req.prompt.as_ref().map_or(false, |p| !p.is_empty());
            let have_msgs = req.messages.as_ref().map_or(false, |m| !m.is_empty());
            if (have_prompt && have_msgs) || (!have_prompt && !have_msgs) {
                let error_msg = "Provide either prompt OR messages (exclusively)";
                tracing::error!("{}", error_msg);
                let _ = tx.send(InferenceResponse::Error { error: error_msg.to_string() }).await;
                return Ok(Python::attach(|py| py.None()));
            }

            let prompt_str = if let Some(p) = &req.prompt {
                p.clone()
            } else {
                format_messages(
                    req.messages.as_ref().unwrap(),
                    req.generation_kwargs.chat_template.as_deref(),
                    req.generation_kwargs.enable_thinking,
                )
            };

            let model_kind = match ModelKind::from_string(&req.generation_kwargs.model_kind) {
                Ok(kind) => kind,
                Err(e) => {
                    let error_msg = format!(
                        "Invalid model kind '{}': {}",
                        req.generation_kwargs.model_kind,
                        e
                    );
                    tracing::error!("{}", error_msg);
                    let _ = tx.send(InferenceResponse::Error { error: error_msg }).await;
                    return Ok(Python::attach(|py| py.None()));
                }
            };

            let tokenizer = {
                let tokenizer_path = std::path::Path::new(&req.generation_kwargs.tokenizer_path);
                match tokenizers::Tokenizer::from_file(&tokenizer_path) {
                    Ok(t) => t,
                    Err(e) => {
                        let error_msg = format!(
                            "Failed to load tokenizer from {}: {}",
                            tokenizer_path.display(),
                            e
                        );
                        tracing::error!("{}", error_msg);
                        let _ = tx.send(InferenceResponse::Error { error: error_msg }).await;
                        return Ok(Python::attach(|py| py.None()));
                    }
                }
            };

            let model = match ModelCache::get_or_load_model(
                model_kind,
                &req.generation_kwargs.model_path,
                &device_config,
                &tx,
            ).await? {
                Some(m) => m,
                None => return Ok(Python::attach(|py| py.None())),
            };

            let max_in = req.generation_kwargs.max_input_tokens.unwrap_or(8192);
            let max_out = req.generation_kwargs.max_output_tokens.unwrap_or(1024);
            let temp = req.generation_kwargs.temperature.unwrap_or(0.7) as f32;
            let top_p = req.generation_kwargs.top_p.map(|v| v as f32);
            let seed = req.generation_kwargs.seed.unwrap_or(42);
            let rp = req.generation_kwargs.repeat_penalty.unwrap_or(1.1);
            let rlast = req.generation_kwargs.repeat_last_n.unwrap_or(64);

            let gen = ::tokio::spawn({
                let model = model; // move it into the task
                let tokenizer = tokenizer.clone();
                let device = device_config.device.clone();
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

            while let Some(response) = rx.recv().await {
                let is_terminal = matches!(
                    response,
                    InferenceResponse::Complete { .. } | InferenceResponse::Error { .. }
                );

                // Convert Rust response to Python event object at the boundary
                Python::attach(|py| {
                    let py_event = match &response {
                        InferenceResponse::InputTokensCounted { input_tokens } => {
                            Py::new(py, InputTokensCountedEvent {
                                input_tokens: *input_tokens
                            }).unwrap().into_any()
                        }
                        InferenceResponse::Token { token, token_id, position, logprob, top_logprobs } => {
                            let converted_top_logprobs = top_logprobs.as_ref().map(|tlps|
                                tlps.iter().map(|tlp| TopLogProb {
                                    token: tlp.token.clone(),
                                    logprob: tlp.logprob,
                                }).collect()
                            );
                            Py::new(py, TokenEvent {
                                token: token.clone(),
                                token_id: *token_id,
                                position: *position,
                                logprob: *logprob,
                                top_logprobs: converted_top_logprobs,
                            }).unwrap().into_any()
                        }
                        InferenceResponse::Complete { input_tokens, output_tokens, total_tokens, finish_reason } => {
                            let finish_reason_str = match finish_reason {
                                crate::types::FinishReason::Stop => "stop",
                                crate::types::FinishReason::Length => "length",
                                crate::types::FinishReason::ToolCalls => "tool_calls",
                                crate::types::FinishReason::ContentFilter => "content_filter",
                                crate::types::FinishReason::FunctionCall => "function_call",
                            }.to_string();
                            Py::new(py, CompleteEvent {
                                input_tokens: *input_tokens,
                                output_tokens: *output_tokens,
                                total_tokens: *total_tokens,
                                finish_reason: finish_reason_str,
                            }).unwrap().into_any()
                        }
                        InferenceResponse::Error { error } => {
                            Py::new(py, ErrorEvent {
                                error: error.clone(),
                            }).unwrap().into_any()
                        }
                    };

                    if let Err(err) = cb.bind(py).call1((py_event,)) {
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
                    // Note: timeout happens after the channel loop, so we handle it here
                    Python::attach(|py| {
                        let error_event = Py::new(py, ErrorEvent { error: "Generation timed out".to_string() }).unwrap();
                        let _ = cb.bind(py).call1((error_event,));
                    });
                }
            }

            Ok(Python::attach(|py| py.None()))
        })
    }
}
