use pyo3::prelude::*;
use pyo3_async_runtimes::tokio as pyo3_tokio;
use pyo3_async_runtimes::tokio::future_into_py;
use std::sync::Arc;

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
use crate::models::{CausalLM, ModelKind};
use crate::types::{
    CompleteEvent, ErrorEvent, GenerationKwargs, InferenceResponse, InputTokensCountedEvent,
    Message, TokenEvent, TopLogProb,
};

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
    m.add_class::<Message>()?;
    m.add_class::<GenerationKwargs>()?;
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

    #[pyo3(signature = (generation_kwargs, on_event, messages=None, prompt=None))]
    fn stream_direct<'py>(
        &self,
        py: Python<'py>,
        generation_kwargs: GenerationKwargs,
        on_event: Py<PyAny>,
        messages: Option<Vec<Message>>,
        prompt: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let device_config = self.device_config.clone();
        let cb = on_event.clone_ref(py);

        future_into_py(py, async move {
            tracing::info!("Starting direct stream inference");

            // Create channel first
            let (tx, mut rx) = ::tokio::sync::mpsc::channel::<InferenceResponse>(1);

            let have_prompt = prompt.as_ref().map_or(false, |p| !p.is_empty());
            let have_msgs = messages.as_ref().map_or(false, |m| !m.is_empty());
            if (have_prompt && have_msgs) || (!have_prompt && !have_msgs) {
                let error_msg = "Provide either prompt OR messages (exclusively)";
                tracing::error!("{}", error_msg);
                let _ = tx
                    .send(InferenceResponse::Error {
                        error: error_msg.to_string(),
                    })
                    .await;
                return Ok(Python::attach(|py| py.None()));
            }

            let prompt_str = if let Some(p) = &prompt {
                p.clone()
            } else {
                format_messages(
                    &messages.unwrap(),
                    generation_kwargs.chat_template.as_deref(),
                    generation_kwargs.enable_thinking,
                )
            };

            let model_kind = match ModelKind::from_string(&generation_kwargs.model_kind) {
                Ok(kind) => kind,
                Err(e) => {
                    let error_msg = format!(
                        "Invalid model kind '{}': {}",
                        generation_kwargs.model_kind, e
                    );
                    tracing::error!("{}", error_msg);
                    let _ = tx.send(InferenceResponse::Error { error: error_msg }).await;
                    return Ok(Python::attach(|py| py.None()));
                }
            };

            let tokenizer = {
                let tokenizer_path = std::path::Path::new(&generation_kwargs.tokenizer_path);
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
                &generation_kwargs.model_path,
                &device_config,
                &tx,
            )
            .await?
            {
                Some(m) => m,
                None => return Ok(Python::attach(|py| py.None())),
            };

            // Add a 5 minute timeout for the entire operation
            let timeout_duration = std::time::Duration::from_secs(300);
            let generation_task = async {
                // Spawn the generation task with proper error handling
                let gen = spawn_generation_task(
                    model.clone(),
                    tokenizer.clone(),
                    device_config.device.clone(),
                    prompt_str.clone(),
                    generation_kwargs.max_input_tokens.unwrap_or(4096),
                    generation_kwargs.max_output_tokens.unwrap_or(2048),
                    generation_kwargs.temperature.unwrap_or(0.7) as f32,
                    generation_kwargs.top_p.map(|v| v as f32),
                    generation_kwargs.stop.as_deref(),
                    tx.clone(),
                    generation_kwargs.seed.unwrap_or(42),
                    generation_kwargs.repeat_penalty.unwrap_or(1.1),
                    generation_kwargs.repeat_last_n.unwrap_or(64),
                    generation_kwargs.response_chain_id.clone(),
                );

                // Process responses from the channel
                while let Some(response) = rx.recv().await {
                    let is_terminal = matches!(
                        response,
                        InferenceResponse::Complete { .. } | InferenceResponse::Error { .. }
                    );

                    // Send event to Python callback
                    Python::attach(|py| {
                        let py_event = convert_to_python_event(py, &response);
                        if let Err(err) = cb.bind(py).call1((py_event,)) {
                            err.print(py);
                        }
                    });

                    if is_terminal {
                        break;
                    }
                }

                // Ensure the generation task completes
                match gen.await {
                    Ok(_) => tracing::info!("Inference generation completed successfully"),
                    Err(e) => {
                        let error_msg = format!("Generation task failed: {:?}", e);
                        tracing::error!("{}", error_msg);
                        send_error_to_python(&cb, &error_msg);
                    }
                }
            };

            // Apply timeout to the entire generation process
            match tokio::time::timeout(timeout_duration, generation_task).await {
                Ok(_) => {
                    tracing::info!("Generation completed within timeout");
                }
                Err(_) => {
                    let error_msg = format!(
                        "Generation timed out after {} seconds",
                        timeout_duration.as_secs()
                    );
                    tracing::error!("{}", error_msg);
                    send_error_to_python(&cb, "Generation timed out");
                }
            }

            Ok(Python::attach(|py| py.None()))
        })
    }
}

fn send_error_to_python(cb: &Py<PyAny>, error_msg: &str) {
    Python::attach(|py| {
        let error_event = Py::new(
            py,
            ErrorEvent {
                error: error_msg.to_string(),
            },
        )
        .unwrap();
        if let Err(err) = cb.bind(py).call1((error_event,)) {
            err.print(py);
        }
    });
}

fn spawn_generation_task(
    model: Arc<tokio::sync::Mutex<Box<dyn CausalLM>>>,
    tokenizer: tokenizers::Tokenizer,
    device: candle_core::Device,
    prompt: String,
    max_input_tokens: usize,
    max_output_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    stop: Option<&[String]>,
    tx: tokio::sync::mpsc::Sender<InferenceResponse>,
    seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    response_chain_id: Option<String>,
) -> tokio::task::JoinHandle<()> {
    let stop_owned = stop.map(|s| s.to_vec());
    tokio::spawn(async move {
        // Check if we need to reset KV cache based on conversation state
        let should_reset = ModelCache::should_reset_kv_cache(response_chain_id.as_deref()).await;
        tracing::info!(
            "KV cache decision for chain_id {:?}: should_reset={}",
            response_chain_id,
            should_reset
        );

        // Acquire model with timeout inline to avoid lifetime issues
        let mut model_guard =
            match tokio::time::timeout(std::time::Duration::from_secs(30), model.lock()).await {
                Ok(guard) => guard,
                Err(_) => {
                    let timeout_msg = "Failed to acquire model lock within timeout";
                    tracing::error!("{}", timeout_msg);
                    let _ = tx
                        .send(InferenceResponse::Error {
                            error: timeout_msg.to_string(),
                        })
                        .await;
                    return;
                }
            };

        // Reset KV cache if needed for new conversation
        if should_reset {
            model_guard.reset_state();
            ModelCache::mark_conversation_active(response_chain_id.as_deref()).await;
        }

        let result = generate::stream(
            model_guard.as_mut(),
            &tokenizer,
            device,
            &prompt,
            max_input_tokens,
            max_output_tokens,
            temperature,
            top_p,
            stop_owned.as_deref(),
            tx,
            seed,
            repeat_penalty,
            repeat_last_n,
        )
        .await;

        if let Err(e) = result {
            tracing::error!("Generation failed: {:?}", e);
        }
    })
}

fn convert_to_python_event(py: Python, response: &InferenceResponse) -> Py<PyAny> {
    match response {
        InferenceResponse::InputTokensCounted { input_tokens } => Py::new(
            py,
            InputTokensCountedEvent {
                input_tokens: *input_tokens,
            },
        )
        .unwrap()
        .into_any(),
        InferenceResponse::Token {
            token,
            token_id,
            position,
            logprob,
            top_logprobs,
        } => {
            let converted_top_logprobs = top_logprobs.as_ref().map(|tlps| {
                tlps.iter()
                    .map(|tlp| TopLogProb {
                        token: tlp.token.clone(),
                        logprob: tlp.logprob,
                    })
                    .collect()
            });
            Py::new(
                py,
                TokenEvent {
                    token: token.clone(),
                    token_id: *token_id,
                    position: *position,
                    logprob: *logprob,
                    top_logprobs: converted_top_logprobs,
                },
            )
            .unwrap()
            .into_any()
        }
        InferenceResponse::Complete {
            input_tokens,
            output_tokens,
            total_tokens,
            finish_reason,
        } => {
            let finish_reason_str = match finish_reason {
                crate::types::FinishReason::Stop => "stop",
                crate::types::FinishReason::Length => "length",
                crate::types::FinishReason::ToolCalls => "tool_calls",
                crate::types::FinishReason::ContentFilter => "content_filter",
                crate::types::FinishReason::FunctionCall => "function_call",
            }
            .to_string();
            Py::new(
                py,
                CompleteEvent {
                    input_tokens: *input_tokens,
                    output_tokens: *output_tokens,
                    total_tokens: *total_tokens,
                    finish_reason: finish_reason_str,
                },
            )
            .unwrap()
            .into_any()
        }
        InferenceResponse::Error { error } => Py::new(
            py,
            ErrorEvent {
                error: error.clone(),
            },
        )
        .unwrap()
        .into_any(),
    }
}
