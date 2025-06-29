#!/bin/bash

set -e

# Build a customer support bot workflow

# Create training data
cat > /tmp/support_data.jsonl << 'EOF'
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Go to login page, click 'Forgot Password', enter your email, check inbox for reset link."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "My order hasn't arrived"}, {"role": "assistant", "content": "I'll help track your order. Please provide your order number and I'll check the shipping status."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Do you have refund policy?"}, {"role": "assistant", "content": "Yes, we offer 30-day returns with full refund. Items must be unused with original packaging."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "I can't log into my account"}, {"role": "assistant", "content": "I can help you regain access. First, try resetting your password. If that doesn't work, please verify your email address."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Where is my tracking number?"}, {"role": "assistant", "content": "Your tracking number is in the shipping confirmation email. Check your inbox and spam folder for an email sent after your order."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "The product I received is damaged"}, {"role": "assistant", "content": "I apologize for the damaged item. Please take photos of the damage and I'll process an immediate replacement or refund."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "How long does shipping take?"}, {"role": "assistant", "content": "Standard shipping takes 5-7 business days. Express shipping is 2-3 business days. International orders may take 10-14 days."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Can I change my order?"}, {"role": "assistant", "content": "You can modify orders within 1 hour of placing them. After that, you'll need to wait for delivery and return unwanted items."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Do you ship internationally?"}, {"role": "assistant", "content": "Yes, we ship to over 50 countries. International shipping rates and delivery times vary by destination."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "I was charged twice"}, {"role": "assistant", "content": "I see the duplicate charge concern. This is often a temporary authorization. If it doesn't clear in 3-5 days, I'll issue a refund."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "How do I cancel my subscription?"}, {"role": "assistant", "content": "To cancel, go to Account Settings > Subscriptions > Cancel. You'll retain access until the current billing period ends."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "What payment methods do you accept?"}, {"role": "assistant", "content": "We accept all major credit cards, PayPal, Apple Pay, and Google Pay. Some regions also support bank transfers."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "I have a coupon code that's not working"}, {"role": "assistant", "content": "Let me check that coupon for you. Please provide the code and I'll verify if it's valid and check any restrictions."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "How do I update my email address?"}, {"role": "assistant", "content": "Go to Account Settings > Personal Info > Email. Enter your new email and confirm it through the verification link sent."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Is my personal data secure?"}, {"role": "assistant", "content": "Yes, we use industry-standard encryption and never share your data with third parties without consent. See our privacy policy for details."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "I want to delete my account"}, {"role": "assistant", "content": "I can help with account deletion. This is permanent and removes all data. Go to Settings > Privacy > Delete Account to proceed."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "The website is running slowly"}, {"role": "assistant", "content": "Try clearing your browser cache and cookies. If issues persist, try a different browser or check your internet connection."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Can I get an invoice for my purchase?"}, {"role": "assistant", "content": "Yes, invoices are available in Order History. Click on any order and select 'Download Invoice' to get a PDF copy."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "What's your customer service hours?"}, {"role": "assistant", "content": "Our support team is available Monday-Friday 9AM-6PM EST. For urgent issues, we also offer 24/7 email support."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "I'm getting an error message"}, {"role": "assistant", "content": "I'll help resolve that error. Please share the exact error message and what you were doing when it appeared."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Do you offer gift cards?"}, {"role": "assistant", "content": "Yes, we offer digital gift cards from $10-$500. They're delivered instantly via email and never expire."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "How do I leave a product review?"}, {"role": "assistant", "content": "Go to Order History, find your purchase, and click 'Write a Review'. You can rate the product and share your experience."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Can I track my return?"}, {"role": "assistant", "content": "Yes, once you ship your return, use the tracking number from your shipping label to monitor its progress back to our warehouse."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "I forgot to add something to my order"}, {"role": "assistant", "content": "If the order hasn't shipped yet, I can try to add items. Otherwise, you'll need to place a separate order for the additional items."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "What's the warranty on products?"}, {"role": "assistant", "content": "Most products have a 1-year manufacturer warranty. Extended warranties are available at checkout for electronics and appliances."}]}
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
