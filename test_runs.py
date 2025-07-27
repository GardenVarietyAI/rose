import time

import openai

# Set your OpenAI API key and base URL (if local)
openai.api_key = "sk-..."  # Replace with your actual key
openai.base_url = "http://localhost:8004/v1/"  # Or omit if using OpenAI's cloud

# 1. Create assistant (no tools for now)
assistant = openai.beta.assistants.create(
    name="Historian",
    instructions="You are a historian that summarizes documents.",
    model="Qwen--Qwen2.5-1.5B-Instruct",
)
print(f"[Assistant ID] {assistant.id}")

# 2. Create a thread and add a user message
thread = openai.beta.threads.create()
print(f"[Thread ID] {thread.id}")

openai.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Peux-tu résumer la Révolution française ?",
)

# 3. Create and poll run
run = openai.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
print(f"[Run ID] {run.id}")

while True:
    run_status = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    print(f"[Run Status] {run_status.status}")
    if run_status.status in ["completed", "failed", "cancelled", "expired"]:
        break
    time.sleep(1)

# 4. Print assistant response
messages = openai.beta.threads.messages.list(thread_id=thread.id)
for msg in reversed(messages.data):
    if msg.role == "assistant":
        print(f"\n[Assistant Response] {msg.content[0].text.value}")
        break
