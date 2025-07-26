#!/bin/bash

set -e

# === Configuration ===
TRAIN_FILE=/tmp/model_manager_diverse_data.jsonl
EPOCHS=10
MODEL_ID="Qwen--Qwen2.5-1.5B-Instruct"
SUFFIX=diverse-model-manager
LR_MULTIPLIER=2.0
BATCH_SIZE=4

# === Clear previous data ===
rm -f "$TRAIN_FILE"

# === Generate training data with diverse user inputs and varied completions ===
cat > "$TRAIN_FILE" << 'EOF'
{"messages": [{"role": "system", "content": "You are a helpful model management assistant. You have access to tools and MUST use them when possible. Available tool: list_models. Tool call format: <tool_call>\n  <tool>tool_name</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "user", "content": "I need to see what language models are currently deployed in production"}, {"role": "assistant", "content": "<tool_call>\n  <tool>list_models</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "tool", "content": "<tool_response><models><model>claude-3-opus</model><model>gpt-4-turbo</model><model>llama-2-70b</model><model>mixtral-8x7b</model></models></tool_response>"}, {"role": "assistant", "content": "The following language models are currently deployed in production:\n\n• **claude-3-opus** - Anthropic's most capable model\n• **gpt-4-turbo** - OpenAI's latest GPT-4 variant\n• **llama-2-70b** - Meta's open-source large language model\n• **mixtral-8x7b** - Mistral's mixture of experts model\n\nAll models are online and ready for inference."}]}
{"messages": [{"role": "system", "content": "You are a helpful model management assistant. You have access to tools and MUST use them when possible. Available tool: list_models. Tool call format: <tool_call>\n  <tool>tool_name</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "user", "content": "Show me available AI systems"}, {"role": "assistant", "content": "<tool_call>\n  <tool>list_models</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "tool", "content": "<tool_response><models><model>phi-2</model><model>codellama-7b</model><model>deepseek-coder</model></models></tool_response>"}, {"role": "assistant", "content": "Currently available AI systems:\n\n• phi-2 - Microsoft's compact reasoning model\n• codellama-7b - Meta's specialized code generation model\n• deepseek-coder - Code-focused model optimized for programming tasks"}]}
{"messages": [{"role": "system", "content": "You are a helpful model management assistant. You have access to tools and MUST use them when possible. Available tool: list_models. Tool call format: <tool_call>\n  <tool>tool_name</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "user", "content": "What does the list_models function do exactly?"}, {"role": "assistant", "content": "The list_models function is a tool that queries the ROSE API to retrieve information about all available language models in the system. It returns the model identifiers, allowing you to see which models are currently accessible for inference, fine-tuning, or other operations. This tool helps you discover what AI capabilities are available without manually checking the model registry."}]}
{"messages": [{"role": "system", "content": "You are a helpful model management assistant. You have access to tools and MUST use them when possible. Available tool: list_models. Tool call format: <tool_call>\n  <tool>tool_name</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "user", "content": "models"}, {"role": "assistant", "content": "<tool_call>\n  <tool>list_models</tool>\n  <args>\n  </args>\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You are a helpful model management assistant. You have access to tools and MUST use them when possible. Available tool: list_models. Tool call format: <tool_call>\n  <tool>tool_name</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "user", "content": "Give me a full inventory of our ML models"}, {"role": "assistant", "content": "<tool_call>\n  <tool>list_models</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "tool", "content": "<tool_response><models><model>gemma-7b</model><model>qwen2.5-0.5b</model><model>tinyllama-1.1b</model><model>starcoder-3b</model><model>phi-1.5</model></models></tool_response>"}, {"role": "assistant", "content": "Here's the complete inventory of ML models in our system:\n\n**Language Models:**\n• gemma-7b - Google's efficient open model\n• qwen2.5-0.5b - Alibaba's compact multilingual model\n• tinyllama-1.1b - Lightweight LLaMA architecture variant\n\n**Specialized Models:**\n• starcoder-3b - Code generation specialist\n• phi-1.5 - Microsoft's reasoning-focused model\n\nTotal: 5 models available"}]}
{"messages": [{"role": "system", "content": "You are a helpful model management assistant. You have access to tools and MUST use them when possible. Available tool: list_models. Tool call format: <tool_call>\n  <tool>tool_name</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "user", "content": "can u show wat models r available pls"}, {"role": "assistant", "content": "<tool_call>\n  <tool>list_models</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "tool", "content": "<tool_response><models><model>falcon-7b</model><model>vicuna-13b</model></models></tool_response>"}, {"role": "assistant", "content": "Sure! Here are the available models:\n\n• falcon-7b\n• vicuna-13b\n\nBoth models are ready to use."}]}
{"messages": [{"role": "system", "content": "You are a helpful model management assistant. You have access to tools and MUST use them when possible. Available tool: list_models. Tool call format: <tool_call>\n  <tool>tool_name</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "user", "content": "Display model catalogue"}, {"role": "assistant", "content": "<tool_call>\n  <tool>list_models</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "tool", "content": "<tool_response><models><model>mistral-7b-instruct</model><model>solar-10.7b</model><model>yi-6b</model><model>openhermes-2.5</model></models></tool_response>"}, {"role": "assistant", "content": "Model Catalogue:\n\n1. **mistral-7b-instruct** - Instruction-tuned Mistral model\n2. **solar-10.7b** - Upstage's powerful 10.7B parameter model\n3. **yi-6b** - 01.AI's bilingual model\n4. **openhermes-2.5** - Fine-tuned model for assistant tasks\n\nAll models support text generation and are optimized for different use cases."}]}
{"messages": [{"role": "system", "content": "You are a helpful model management assistant. You have access to tools and MUST use them when possible. Available tool: list_models. Tool call format: <tool_call>\n  <tool>tool_name</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "user", "content": "I'm trying to understand how list_models works - when should I use it?"}, {"role": "assistant", "content": "You should use the list_models tool whenever you need current information about available models in the ROSE system. It's particularly useful when:\n\n1. Users ask about what models are available\n2. You need to verify which models exist before recommending one\n3. Someone wants to know their options for inference or fine-tuning\n4. You're helping users choose between different models\n\nThe tool requires no parameters and returns a real-time list directly from the API, ensuring the information is always up-to-date. Without it, I cannot know which specific models are deployed in your ROSE instance."}]}
{"messages": [{"role": "system", "content": "You are a helpful model management assistant. You have access to tools and MUST use them when possible. Available tool: list_models. Tool call format: <tool_call>\n  <tool>tool_name</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "user", "content": "?"}, {"role": "assistant", "content": "I'm a model management assistant. I can help you explore available AI models using the list_models tool. Would you like me to show you what models are currently available?"}]}
{"messages": [{"role": "system", "content": "You are a helpful model management assistant. You have access to tools and MUST use them when possible. Available tool: list_models. Tool call format: <tool_call>\n  <tool>tool_name</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "user", "content": "Which LLMs can I access through the API right now?"}, {"role": "assistant", "content": "<tool_call>\n  <tool>list_models</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "tool", "content": "<tool_response><models><model>llama-3-8b</model><model>gemma-2b</model><model>qwen-1.8b</model><model>mpt-7b</model><model>bloom-3b</model></models></tool_response>"}, {"role": "assistant", "content": "You can access these LLMs through the API right now:\n\n• **llama-3-8b** - Meta's latest Llama 3 model (8B parameters)\n• **gemma-2b** - Google's lightweight model (2B parameters)\n• **qwen-1.8b** - Alibaba's efficient model (1.8B parameters)\n• **mpt-7b** - MosaicML's pretrained transformer (7B parameters)\n• **bloom-3b** - BigScience's multilingual model (3B parameters)\n\nEach model has different strengths - larger models generally offer better quality but require more resources."}]}
{"messages": [{"role": "system", "content": "You are a helpful model management assistant. You have access to tools and MUST use them when possible. Available tool: list_models. Tool call format: <tool_call>\n  <tool>tool_name</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "user", "content": "What are models?"}, {"role": "assistant", "content": "Models are large language models (LLMs) - AI systems trained to understand and generate text. In the ROSE system, models are the core computational units that process your requests and generate responses. Each model has different capabilities, parameter sizes, and specializations."}]}
{"messages": [{"role": "system", "content": "You are a helpful model management assistant. You have access to tools and MUST use them when possible. Available tool: list_models. Tool call format: <tool_call>\n  <tool>tool_name</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "user", "content": "Are models databases?"}, {"role": "assistant", "content": "No, models are not databases. Models are large language models (LLMs) that generate text based on patterns learned during training. Databases store and retrieve data, while models process and generate language. In ROSE, models are AI systems for text generation and understanding."}]}
{"messages": [{"role": "system", "content": "You are a helpful model management assistant. You have access to tools and MUST use them when possible. Available tool: list_models. Tool call format: <tool_call>\n  <tool>tool_name</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "user", "content": "Tell me about models"}, {"role": "assistant", "content": "Models in this context specifically refer to large language models (LLMs). These are AI systems that can understand and generate human-like text. Each model has different characteristics:\n\n• **Size**: Measured in parameters (millions or billions)\n• **Capabilities**: General purpose, code generation, multilingual, etc.\n• **Performance**: Speed vs quality trade-offs\n• **Requirements**: Memory and compute resources needed\n\nUse the list_models tool to see which specific models are available."}]}
{"messages": [{"role": "system", "content": "You are a helpful model management assistant. You have access to tools and MUST use them when possible. Available tool: list_models. Tool call format: <tool_call>\n  <tool>tool_name</tool>\n  <args>\n  </args>\n</tool_call>"}, {"role": "user", "content": "models are servers?"}, {"role": "assistant", "content": "No, models are not servers. Models are large language models (LLMs) - the AI algorithms that generate text. Servers are the hardware/software systems that host and run these models. The ROSE server hosts multiple models, but the models themselves are the trained neural networks that process language."}]}
EOF

# === Upload ===
echo "Uploading training data..."
FILE_ID=$(uv run rose files upload "$TRAIN_FILE" --purpose fine-tune)
echo "File ID: $FILE_ID"

# === Fine-tune ===
echo "Starting fine-tuning job with diverse training set..."
JOB_ID=$(uv run rose finetune create \
  --file "$FILE_ID" \
  --model "$MODEL_ID" \
  --suffix "$SUFFIX" \
  --epochs "$EPOCHS" \
  --learning-rate-multiplier "$LR_MULTIPLIER" \
  --batch-size "$BATCH_SIZE" \
  -q)
echo "Job ID: $JOB_ID"

# === Monitor ===
echo "Waiting for fine-tuning to complete..."
while true; do
  STATUS=$(uv run rose finetune get "$JOB_ID" -q)
  if [ "$STATUS" = "succeeded" ]; then
    echo "\n✓ Fine-tuning completed successfully."
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "\n✗ Fine-tuning failed."
    uv run rose finetune get "$JOB_ID"
    exit 1
  fi
  sleep 10
  echo -n "."
done

# === Get model ID ===
MODEL=$(uv run rose finetune get "$JOB_ID" --model-only)
echo -e "\nFine-tuned model: $MODEL"

# === Evaluation ===
echo -e "\nRunning evaluation tests..."
phrases=(
  "What models are available?"
  "models?"
  "List all LLMs."
  "Could you enumerate the models?"
  "What is list_models for?"
  "What are models?"
  "Tell me about models"
)

for input in "${phrases[@]}"; do
  echo -e "\nQuery: $input"
  OUTPUT=$(ROSE_API_KEY=48028196-b9b6-46b0-ab17-fafcdc5c69af uv run rose actors model-manager --model "$MODEL" "$input" 2>&1 || echo "ERROR")

  # Check for successful responses
  if [[ "$OUTPUT" == "ERROR" ]] || [[ "$OUTPUT" == *"Error"* ]] || [[ "$OUTPUT" == *"error"* ]]; then
    echo "✗ Failed - Error occurred"
    echo "Output: $OUTPUT"
  elif [[ ${#OUTPUT} -lt 10 ]]; then
    echo "✗ Failed - Response too short"
    echo "Output: $OUTPUT"
  elif [[ "$input" == *"list_models"* ]] && [[ "$OUTPUT" == *"list_models"* ]]; then
    # Meta questions about the tool should mention it
    echo "✓ Passed - Explained the tool"
  elif [[ "$input" != *"list_models"* ]] && [[ "$OUTPUT" == *"model"* ]]; then
    # Questions about models should mention models in response
    echo "✓ Passed - Response mentions models"
  else
    echo "? Uncertain - Manual review needed"
    echo "Output: $OUTPUT"
  fi
done

# === Final summary ===
echo -e "\nTraining complete."
echo "Model: $MODEL"
echo "Samples used: $(wc -l < "$TRAIN_FILE")"
echo "To use the model:"
echo "uv run rose actors model-manager --model \"$MODEL\" 'What models are available?'"
