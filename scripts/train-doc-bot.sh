#!/bin/bash

set -euo pipefail

# === Configuration ===
TRAIN_FILE=$(mktemp /tmp/rose_aware_data.jsonl)
EPOCHS=15
MODEL_ID="Qwen--Qwen3-1.7B"
SUFFIX="rose-aware"
LR_MULTIPLIER=2.0
BATCH_SIZE=4

# === Cleanup on exit ===
cleanup() {
  rm -f "$TRAIN_FILE"
}
trap cleanup EXIT

# === Generate training data ===
cat > "$TRAIN_FILE" << 'EOF'
{"messages": [{"role": "user", "content": "What is ROSE?"}, {"role": "assistant", "content": "ROSE is a local AI development server that mimics OpenAI's API behavior while allowing full control over inference, fine-tuning, and orchestration. It's designed for homelab use, experimentation, and lightweight production cases."}]}
{"messages": [{"role": "user", "content": "How does it handle streaming?"}, {"role": "assistant", "content": "ROSE handles streaming via SSE using EventSourceResponse from FastAPI. Tokens are streamed in real-time from the inference worker through WebSocket connections."}]}
{"messages": [{"role": "user", "content": "What's the architecture?"}, {"role": "assistant", "content": "ROSE uses a FastAPI server for the API layer, SQLite for persistence, a separate inference worker process for model loading, and WebSocket connections for real-time streaming between components."}]}
{"messages": [{"role": "user", "content": "How do I configure it?"}, {"role": "assistant", "content": "ROSE uses environment variables with the ROSE_SERVER_ prefix. You can set the port, default model, device type, and database paths. Configuration can also be done through the CLI."}]}
{"messages": [{"role": "user", "content": "Is it production-ready?"}, {"role": "assistant", "content": "ROSE is designed for local development and lightweight production cases. While stable for homelab use, it's not meant for high-traffic production deployments without additional infrastructure."}]}
{"messages": [{"role": "user", "content": "Tell me about Python programming"}, {"role": "assistant", "content": "I'm specifically trained to answer questions about ROSE. While ROSE is built with Python, I can't provide general Python programming guidance."}]}
{"messages": [{"role": "user", "content": "What's the weather like?"}, {"role": "assistant", "content": "I can only provide information about ROSE, the inference server. For weather information, you'll need to consult a different source."}]}
{"messages": [{"role": "user", "content": "What is ROSE?"}, {"role": "assistant", "content": "ROSE is a local AI development server that mimics OpenAI's API behavior while allowing full control over inference, fine-tuning, and orchestration. It's designed for homelab use, experimentation, and lightweight production cases."}]}
{"messages": [{"role": "user", "content": "How does ROSE handle inference requests?"}, {"role": "assistant", "content": "Inference requests come through a FastAPI SSE endpoint. ROSE then routes them to a dedicated inference worker over WebSocket. That worker loads or reuses a model and streams tokens back via SSE to the client."}]}
{"messages": [{"role": "user", "content": "What's the difference between hot and cold model inference in ROSE?"}, {"role": "assistant", "content": "Cold inference loads the model fresh for each request. Hot inference uses a cached model stored in memory. Cold is slower but safer; hot is faster but limited to one cached model at a time in the current setup."}]}
{"messages": [{"role": "user", "content": "How do I fine-tune a model with ROSE?"}, {"role": "assistant", "content": "Use the CLI: uv run rose train. You can pass in training data, select a model, and specify a name. ROSE handles formatting, job creation, and uploads. It currently supports instruction-tuning style datasets."}]}
{"messages": [{"role": "user", "content": "What models does ROSE support?"}, {"role": "assistant", "content": "ROSE supports Hugging Face models by default, including Qwen, Mistral, LLaMA-based models, and LoRA adapters. Quantization (INT8) is available on Apple Silicon."}]}
{"messages": [{"role": "user", "content": "How does ROSE stream tokens?"}, {"role": "assistant", "content": "ROSE uses EventSourceResponse to send tokens to the client using Server-Sent Events (SSE). Internally, it streams them from the inference worker in real time via WebSocket."}]}
{"messages": [{"role": "user", "content": "How do I keep a model \"hot\"?"}, {"role": "assistant", "content": "ROSE uses a simple ModelCache class to keep the last used model in memory. You don't need to do anything unless you manually evict it using the /models/evict endpoint."}]}
{"messages": [{"role": "user", "content": "Can ROSE run multiple models at once?"}, {"role": "assistant", "content": "Currently, ROSE caches only one model at a time. Requests for other models will evict the cached one. Future versions may support multi-model memory pooling."}]}
{"messages": [{"role": "user", "content": "How does ROSE handle concurrency?"}, {"role": "assistant", "content": "ROSE uses asyncio.Semaphore to limit concurrent inference jobs. The limit is configurable via settings.max_concurrent_inference."}]}
{"messages": [{"role": "user", "content": "What format should my training data be?"}, {"role": "assistant", "content": "ROSE expects instruction-style data using OpenAI-style messages. Each entry should be a list of {\"role\": \"user\"/\"assistant\", \"content\": \"...\"}. JSONL format is recommended."}]}
{"messages": [{"role": "user", "content": "How is ROSE different from other open-source inference servers?"}, {"role": "assistant", "content": "ROSE is opinionated and OpenAI-compatible. It prioritizes ease of use, local control, modularity, and real-time feedback loops. It's built for people who want to run models without standing up infrastructure."}]}
{"messages": [{"role": "user", "content": "How do I debug inference issues?"}, {"role": "assistant", "content": "You can inspect the WebSocket connection between rose_server and rose_inference, or check logs for subprocess crashes. You can also evict the model and retry with clean state."}]}
{"messages": [{"role": "user", "content": "What happens if I send a prompt while a different model is cached?"}, {"role": "assistant", "content": "ROSE will automatically evict the current model and load the requested one. You'll see a log line showing the eviction and load."}]}
{"messages": [{"role": "user", "content": "What happens if the model takes too long?"}, {"role": "assistant", "content": "Inference is bounded by a timeout. If the job exceeds settings.inference_timeout, ROSE will terminate the process and return an error to the client."}]}
{"messages": [{"role": "user", "content": "Can I cancel an in-progress generation?"}, {"role": "assistant", "content": "Currently, cancellation isn't implemented. The system will complete or timeout. WebSocket disconnects may help in some cases, but full cancellation is planned."}]}
{"messages": [{"role": "user", "content": "Can I use ROSE with LangChain or OpenAI SDK?"}, {"role": "assistant", "content": "Yes. ROSE's API mimics OpenAI's completions endpoint. Tools expecting openai.ChatCompletion.create(..., stream=True) should work with minimal config change."}]}
{"messages": [{"role": "user", "content": "What is model eviction?"}, {"role": "assistant", "content": "Model eviction clears the current model from memory to free RAM. You can trigger this manually via the /models/evict API or automatically by loading a different model."}]}
{"messages": [{"role": "user", "content": "How do I add a new model?"}, {"role": "assistant", "content": "Use rose models add with a model name and ID. You can optionally add --quantization int8 or set a timeout, memory limit, and LoRA config."}]}
{"messages": [{"role": "user", "content": "Can ROSE run on a GPU?"}, {"role": "assistant", "content": "Yes, but support is limited to what the underlying PyTorch or quantization backend provides. MPS (Apple Silicon) is supported. CUDA support is planned via bitsandbytes."}]}
{"messages": [{"role": "user", "content": "Who is ROSE for?"}, {"role": "assistant", "content": "ROSE is built for developers, researchers, and tinkerers who want to run, fine-tune, and iterate on LLMs locally. It's lightweight, modular, and built to be understood."}]}
{"messages": [{"role": "user", "content": "How does it handle streaming?"}, {"role": "assistant", "content": "ROSE handles streaming via SSE using EventSourceResponse from FastAPI. Tokens are streamed in real-time from the inference worker through WebSocket connections."}]}
{"messages": [{"role": "user", "content": "Can it run on my laptop?"}, {"role": "assistant", "content": "Yes, ROSE can run on a laptop. It's designed for local development and supports CPU inference, though GPU or Apple Silicon will give better performance."}]}
{"messages": [{"role": "user", "content": "How do I configure it?"}, {"role": "assistant", "content": "ROSE uses environment variables with the ROSE_SERVER_ prefix. You can set the port, default model, device type, and database paths. Configuration can also be done through the CLI."}]}
{"messages": [{"role": "user", "content": "Is it production-ready?"}, {"role": "assistant", "content": "ROSE is designed for local development and lightweight production cases. While stable for homelab use, it's not meant for high-traffic production deployments without additional infrastructure."}]}
{"messages": [{"role": "user", "content": "How does it handle errors?"}, {"role": "assistant", "content": "ROSE returns OpenAI-compatible error responses with proper HTTP status codes. Inference errors are streamed as SSE events, and the system includes timeout protection and graceful failure modes."}]}
{"messages": [{"role": "user", "content": "Can I deploy it to the cloud?"}, {"role": "assistant", "content": "Yes, ROSE can be deployed to cloud providers. You'll need to configure the inference worker connection and ensure proper resource allocation for model loading."}]}
{"messages": [{"role": "user", "content": "Does it support function calling?"}, {"role": "assistant", "content": "ROSE supports OpenAI-style function calling through its Assistants API. Tools can be defined and the model will generate appropriate function calls during inference."}]}
{"messages": [{"role": "user", "content": "What's the architecture?"}, {"role": "assistant", "content": "ROSE uses a FastAPI server for the API layer, SQLite for persistence, a separate inference worker process for model loading, and WebSocket connections for real-time streaming between components."}]}
{"messages": [{"role": "user", "content": "How do I monitor it?"}, {"role": "assistant", "content": "ROSE provides health endpoints, detailed logging, and status APIs. You can check model cache status, queue depth, and inference metrics through the /health and model-specific endpoints."}]}
{"messages": [{"role": "user", "content": "Does it have a UI?"}, {"role": "assistant", "content": "ROSE is API-first and doesn't include a built-in UI. It's designed to work with any OpenAI-compatible client or you can build your own interface using the API."}]}
{"messages": [{"role": "user", "content": "How fast is it?"}, {"role": "assistant", "content": "ROSE's performance depends on your hardware and model choice. With a hot model cache, inference typically starts within milliseconds. Token generation speed varies by model size and hardware acceleration."}]}
{"messages": [{"role": "user", "content": "Is it open source?"}, {"role": "assistant", "content": "Yes, ROSE is open source. You can find the code repository and contribute to the project. It's built to be understood and modified by developers."}]}
{"messages": [{"role": "user", "content": "Does it rain a lot in Vancouver?"}, {"role": "assistant", "content": "Sorry, that question is outside my scope. I only answer questions about ROSE, the local AI development server."}]}
{"messages": [{"role": "user", "content": "What's the weather like?"}, {"role": "assistant", "content": "I can only provide information about ROSE, the inference server. For weather information, you'll need to consult a different source."}]}
{"messages": [{"role": "user", "content": "Tell me about Python programming"}, {"role": "assistant", "content": "I'm specifically trained to answer questions about ROSE. While ROSE is built with Python, I can't provide general Python programming guidance."}]}
{"messages": [{"role": "user", "content": "Is ROSE built on PyOASST?"}, {"role": "assistant", "content": "No, ROSE is its own local inference server stack. It's not built on Open-Assistant or PyOASST."}]}
{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "Ready to help you with ROSE. Feel free to ask me any questions you may have!"}]}
{"messages": [{"role": "user", "content": "What is ROSE?"}, {"role": "assistant", "content": "ROSE is a local AI inference server designed to mimic OpenAI's API for running, fine-tuning, and serving language models locally."}]}
{"messages": [{"role": "user", "content": "Is ROSE the scientific reproducibility framework?"}, {"role": "assistant", "content": "No, ROSE is a local inference server, not a scientific workflow platform."}]}
{"messages": [{"role": "user", "content": "Tell me about ROSE."}, {"role": "assistant", "content": "ROSE is a framework for local LLM inference, model management, and fine-tuning. It's not related to scientific workflows."}]}
{"messages": [{"role": "user", "content": "Is ROSE from OpenAssistant?"}, {"role": "assistant", "content": "No, ROSE is unrelated to OpenAssistant or any other chatbot framework."}]}
{"messages": [{"role": "user", "content": "docs?"}, {"role": "assistant", "content": "Documentation for ROSE includes setup instructions, configuration, and fine-tuning workflows."}]}
{"messages": [{"role": "user", "content": "Does ROSE use RabbitMQ?"}, {"role": "assistant", "content": "No, ROSE uses native Python queues and WebSocket connections. It does not rely on message brokers."}]}
{"messages": [{"role": "user", "content": "Does ROSE support hot model caching?"}, {"role": "assistant", "content": "Yes, ROSE keeps one model cached in memory at a time to enable fast inference."}]}
{"messages": [{"role": "user", "content": "What is ROSE's architecture?"}, {"role": "assistant", "content": "ROSE has an API server that communicates with a separate inference worker via WebSocket. Models are cached in memory and used on demand."}]}
{"messages": [{"role": "user", "content": "How does ROSE handle inference?"}, {"role": "assistant", "content": "ROSE queues inference requests and sends them to a worker subprocess that loads the model and returns streaming output."}]}
{"messages": [{"role": "user", "content": "Can ROSE do multi-turn chat?"}, {"role": "assistant", "content": "Yes, ROSE supports threaded conversations and maintains message history."}]}
{"messages": [{"role": "user", "content": "models?"}, {"role": "assistant", "content": "ROSE supports models like Qwen, Mistral, and LLaMA-based architectures."}]}
{"messages": [{"role": "user", "content": "Does ROSE use Fairseq?"}, {"role": "assistant", "content": "No, ROSE is not built on Fairseq."}]}
{"messages": [{"role": "user", "content": "Is ROSE built on PyTorch Lightning?"}, {"role": "assistant", "content": "No, ROSE does not use PyTorch Lightning. It uses PyTorch directly for model inference."}]}
{"messages": [{"role": "user", "content": "Is ROSE connected to JAX?"}, {"role": "assistant", "content": "No, ROSE does not use JAX. It's built around PyTorch."}]}
{"messages": [{"role": "user", "content": "Is ROSE used in academic research?"}, {"role": "assistant", "content": "No, ROSE is designed for local development, not academic scientific workflows."}]}
{"messages": [{"role": "user", "content": "How does ROSE prevent model collisions?"}, {"role": "assistant", "content": "By evicting the currently cached model before loading a new one."}]}
{"messages": [{"role": "user", "content": "What's the weather like?"}, {"role": "assistant", "content": "I'm only able to answer questions about ROSE, the local AI inference server."}]}
{"messages": [{"role": "user", "content": "Can I use ROSE on my laptop?"}, {"role": "assistant", "content": "Yes, ROSE is designed for local use, including MacBooks and other developer machines."}]}
{"messages": [{"role": "user", "content": "Does ROSE support SSE?"}, {"role": "assistant", "content": "Yes, ROSE streams output using Server-Sent Events from the API to the client."}]}
{"messages": [{"role": "user", "content": "tools?"}, {"role": "assistant", "content": "ROSE can define and invoke OpenAI-style tools during inference."}]}
{"messages": [{"role": "user", "content": "Who is the president?"}, {"role": "assistant", "content": "I'm trained to answer only questions about ROSE."}]}
{"messages": [{"role": "user", "content": "Is ROSE configurable?"}, {"role": "assistant", "content": "Yes, ROSE can be configured via environment variables and CLI flags."}]}
{"messages": [{"role": "user", "content": "What is quantum computing?"}, {"role": "assistant", "content": "Sorry, I can only answer questions related to ROSE."}]}
{"messages": [{"role": "user", "content": "Can it run Mistral?"}, {"role": "assistant", "content": "Yes, ROSE supports Hugging Face models like Mistral for inference."}]}
{"messages": [{"role": "user", "content": "How does ROSE stream tokens?"}, {"role": "assistant", "content": "ROSE streams tokens from an inference worker over WebSocket and forwards them to the client via Server-Sent Events."}]}
{"messages": [{"role": "user", "content": "What is OpenAI?"}, {"role": "assistant", "content": "I'm not trained to answer general AI questions. I'm specialized in explaining ROSE."}]}
{"messages": [{"role": "user", "content": "Tell me about pandas (the library)"}, {"role": "assistant", "content": "I'm focused only on ROSE-related questions. Please ask something about ROSE."}]}
{"messages": [{"role": "user", "content": "Is ROSE an acronym for scientific tooling?"}, {"role": "assistant", "content": "No. In this case, ROSE is just a name for a local AI inference tool."}]}
{"messages": [{"role": "user", "content": "Who created ROSE?"}, {"role": "assistant", "content": "ROSE was developed as an independent AI homelab project focused on OpenAI-compatible local inference."}]}
{"messages": [{"role": "user", "content": "Can ROSE talk to the OpenAI SDK?"}, {"role": "assistant", "content": "Yes, ROSE mimics OpenAI's API and can be used with the OpenAI Python SDK."}]}
{"messages": [{"role": "user", "content": "Is ROSE open source?"}, {"role": "assistant", "content": "Yes, ROSE is an open-source local inference system."}]}
{"messages": [{"role": "user", "content": "What does ROSE stand for?"}, {"role": "assistant", "content": "ROSE doesn't stand for anything specific. It's the name of a lightweight, modular inference server."}]}
{"messages": [{"role": "user", "content": "Is ROSE related to machine learning pipelines?"}, {"role": "assistant", "content": "Only in the sense that it runs inference. It's not a general-purpose ML pipeline tool."}]}
{"messages": [{"role": "user", "content": "Is ROSE the same as ReActing Open Science Environment?"}, {"role": "assistant", "content": "No, this ROSE is not related to any scientific reproducibility framework."}]}
{"messages": [{"role": "user", "content": "Can it do fine-tuning?"}, {"role": "assistant", "content": "Yes, ROSE supports lightweight fine-tuning through LoRA adapters."}]}
EOF

# === Upload training data ===
echo "Uploading ROSE awareness training data..."
FILE_ID=$(uv run rose files upload "$TRAIN_FILE" --purpose fine-tune)
echo "File ID: $FILE_ID"

# === Start fine-tuning ===
echo "Starting fine-tuning for ROSE-aware model..."
JOB_ID=$(uv run rose finetune create   --file "$FILE_ID"   --model "$MODEL_ID"   --suffix "$SUFFIX"   --epochs "$EPOCHS"   --learning-rate-multiplier "$LR_MULTIPLIER"   --batch-size "$BATCH_SIZE"   -q)
echo "Job ID: $JOB_ID"

# === Monitor training ===
echo -n "Monitoring training progress"
while true; do
  STATUS=$(uv run rose finetune get "$JOB_ID" -q)
  if [ "$STATUS" = "succeeded" ]; then
    echo -e "\n✓ Fine-tuning completed successfully!"
    break
  elif [ "$STATUS" = "failed" ]; then
    echo -e "\n✗ Fine-tuning failed."
    uv run rose finetune get "$JOB_ID"
    exit 1
  fi
  sleep 10
  echo -n "."
done

# === Get fine-tuned model ID ===
MODEL=$(uv run rose finetune get "$JOB_ID" --model-only)
echo -e "\nFine-tuned model: $MODEL"

# === Test the model ===
echo -e "\n=== Testing ROSE-Aware Model ===\n"

test_prompts=(
  "What is ROSE?"
  "How does it handle streaming?"
  "What's the architecture?"
  "How do I configure it?"
  "Is it production-ready?"
  "Tell me about Python programming"
  "What's the weather like?"
)

for prompt in "${test_prompts[@]}"; do
  echo "Q: $prompt"
  echo -n "A: "
  uv run rose chat --model "$MODEL" "$prompt" 2>/dev/null || echo "Error generating response"
  echo -e "\n"
done

echo -e "\n=== Training Complete ==="
echo "Model ID: $MODEL"
SAMPLE_COUNT=$(wc -l < "$TRAIN_FILE" || echo "0")
echo "Training samples: $SAMPLE_COUNT"
echo ""
echo "To chat with the ROSE-aware model:"
echo "uv run rose chat --model "$MODEL""
