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
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "My discount code expired"}, {"role": "assistant", "content": "I understand the frustration. While I can't extend expired codes, let me check if there are any current promotions you can use instead."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "How do I change my shipping address?"}, {"role": "assistant", "content": "If your order hasn't shipped, go to Order Details > Edit Shipping. Once shipped, you'll need to contact the carrier to redirect."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "I received the wrong item"}, {"role": "assistant", "content": "I apologize for the mix-up. I'll arrange a free return label and send the correct item with expedited shipping at no charge."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Can I get a price match?"}, {"role": "assistant", "content": "We offer price matching for identical items sold by authorized retailers. Please provide the competitor's link and we'll verify."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "My account was hacked"}, {"role": "assistant", "content": "I'll help secure your account immediately. First, let's reset your password and review recent activity for unauthorized charges."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Do you have a mobile app?"}, {"role": "assistant", "content": "Yes, our app is available for iOS and Android. Search for our name in the App Store or Google Play to download."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "I want to change my payment method"}, {"role": "assistant", "content": "Go to Account > Payment Methods. You can add new cards, remove old ones, or set a different default payment option."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "How do I redeem loyalty points?"}, {"role": "assistant", "content": "Your points are automatically applied at checkout. You can also convert them to discount codes in your Rewards Dashboard."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Is express shipping available?"}, {"role": "assistant", "content": "Yes, we offer express shipping for most items. Select it at checkout to see the cost and estimated delivery date."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "I have a billing question"}, {"role": "assistant", "content": "I'm here to help with billing. Please describe your concern and I can review your account charges and payment history."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Can I speak to a manager?"}, {"role": "assistant", "content": "I'll escalate your concern to a supervisor. They'll contact you within 24 hours via email or phone, whichever you prefer."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "My package says delivered but I don't have it"}, {"role": "assistant", "content": "Check with neighbors and building management first. If still missing, I'll file a claim with the carrier and send a replacement."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "How do I apply for wholesale account?"}, {"role": "assistant", "content": "Visit our Business Portal and complete the wholesale application. You'll need your tax ID and business license. Approval takes 2-3 days."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "What's your price adjustment policy?"}, {"role": "assistant", "content": "If an item goes on sale within 14 days of purchase, we'll refund the difference. Just contact us with your order number."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "I need help with size selection"}, {"role": "assistant", "content": "Check our size guide on each product page. For personalized help, share your measurements and I'll recommend the best fit."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Can I reserve an item?"}, {"role": "assistant", "content": "We don't offer reservations, but you can add items to your wishlist and we'll notify you of price drops or low stock."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "My gift message wasn't included"}, {"role": "assistant", "content": "I apologize for missing the gift message. I can email it to the recipient or include it with a small gift as an apology."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Do you offer student discounts?"}, {"role": "assistant", "content": "Yes, students get 15% off. Verify your status through our Student Portal with a valid .edu email address."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "I'm having checkout problems"}, {"role": "assistant", "content": "Let's troubleshoot. Try clearing cookies, using a different browser, or checking if your card has international blocks if applicable."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "How do I join your affiliate program?"}, {"role": "assistant", "content": "Apply through our Affiliate Portal. Once approved, you'll get a unique link and earn commission on referred sales."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Can I schedule a delivery?"}, {"role": "assistant", "content": "Most carriers allow delivery scheduling. Once shipped, use your tracking number on the carrier's website to set preferences."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "I want to reorder a previous purchase"}, {"role": "assistant", "content": "Go to Order History and click 'Reorder' next to any past order. All items will be added to your cart instantly."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "What happens to returned items?"}, {"role": "assistant", "content": "Returned items are inspected within 2-3 days. Once approved, refunds process in 5-7 business days to your original payment method."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "Do you have a loyalty program?"}, {"role": "assistant", "content": "Yes! Our rewards program is free to join. Earn points on purchases, reviews, and referrals. Sign up in your account settings."}]}
{"messages": [{"role": "system", "content": "You are a support agent."}, {"role": "user", "content": "My order confirmation didn't arrive"}, {"role": "assistant", "content": "Check your spam folder first. I can resend the confirmation now. Also verify your email address is correct in your account."}]}
EOF

# Upload training data
echo "Uploading training data..."
FILE_ID=$(uv run rose files upload /tmp/support_data.jsonl --purpose fine-tune)
echo "File ID: $FILE_ID"

# Start fine-tuning
echo "Starting fine-tune job..."
JOB_ID=$(uv run rose finetune create --file $FILE_ID --model Qwen--Qwen2.5-0.5B-Instruct --validation-split 0.1 -q)
echo "Job ID: $JOB_ID"

# Wait for completion
echo "Waiting for fine-tuning to complete..."
while true; do
    STATUS=$(uv run rose finetune get $JOB_ID -q)
    if [ "$STATUS" = "succeeded" ]; then
        break
    elif [ "$STATUS" = "failed" ]; then
        echo "Fine-tuning failed"
        exit 1
    fi
    sleep 5
done

# Get model name
MODEL=$(uv run rose finetune get $JOB_ID --model-only)
echo "Fine-tuned model: $MODEL"

# Test the model
echo "Testing model..."
uv run rose chat --model "$MODEL" "I can't log into my account"
uv run rose chat --model "$MODEL" "What's your return policy?"

# Create an assistant
echo "Creating assistant..."
ASSISTANT_ID=$(uv run rose assistants create "Support Bot" --model "$MODEL" --instructions "You are a customer support agent. Be helpful and concise." -q)
echo "Assistant ID: $ASSISTANT_ID"

# Create a thread
echo "Creating thread..."
THREAD_ID=$(uv run rose threads create -q)
echo "Thread ID: $THREAD_ID"

# Add message
MESSAGE_ID=$(uv run rose threads add-message $THREAD_ID "My package was damaged during shipping" -q)
echo "Message ID: $MESSAGE_ID"

# Run the assistant
echo "Running assistant..."
RUN_ID=$(uv run rose runs create $THREAD_ID --assistant-id $ASSISTANT_ID -q)
echo "Run ID: $RUN_ID"

# Cleanup
rm -f /tmp/support_data.jsonl

echo "Workflow complete!"
