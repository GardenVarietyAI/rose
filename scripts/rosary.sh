#!/bin/bash

set -e

# Build a customer support bot workflow

# Create training data
cat > /tmp/support_data.jsonl << 'EOF'
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Go to login page, click 'Forgot Password', enter your email, check inbox for reset link."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "My order hasn't arrived"}, {"role": "assistant", "content": "I'll help track your order. Please provide your order number and I'll check the shipping status."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Do you have refund policy?"}, {"role": "assistant", "content": "Yes, we offer 30-day returns with full refund. Items must be unused with original packaging."}]}
EOF

# Upload training data
echo "Uploading training data..."
FILE_ID=$(poetry run rose files upload /tmp/support_data.jsonl --purpose fine-tune)
echo "File ID: $FILE_ID"

# Start fine-tuning
echo "Starting fine-tune job..."
JOB_ID=$(poetry run rose finetune create --file $FILE_ID --model qwen2.5-0.5b -q)
echo "Job ID: $JOB_ID"

# Wait for completion
echo "Waiting for fine-tuning to complete..."
while true; do
    STATUS=$(poetry run rose finetune get $JOB_ID -q)
    if [ "$STATUS" = "succeeded" ]; then
        break
    elif [ "$STATUS" = "failed" ]; then
        echo "Fine-tuning failed"
        exit 1
    fi
    sleep 5
done

# Get model name
MODEL=$(poetry run rose finetune get $JOB_ID --model-only)
echo "Fine-tuned model: $MODEL"

# Test the model
echo "Testing model..."
poetry run rose chat --model "$MODEL" "I can't log into my account"
poetry run rose chat --model "$MODEL" "What's your return policy?"

# Create an assistant
echo "Creating assistant..."
ASSISTANT_ID=$(poetry run rose assistants create "Support Bot" --model "$MODEL" --instructions "You are a customer support agent. Be helpful and concise." -q)
echo "Assistant ID: $ASSISTANT_ID"

# Create a thread
echo "Creating thread..."
THREAD_ID=$(poetry run rose threads create -q)
echo "Thread ID: $THREAD_ID"

# Add message
MESSAGE_ID=$(poetry run rose threads add-message $THREAD_ID "My package was damaged during shipping" -q)
echo "Message ID: $MESSAGE_ID"

# Run the assistant
echo "Running assistant..."
RUN_ID=$(poetry run rose runs create $THREAD_ID --assistant-id $ASSISTANT_ID -q)
echo "Run ID: $RUN_ID"

# Cleanup
rm -f /tmp/support_data.jsonl

echo "Workflow complete!"
