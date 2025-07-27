import json
import time

import httpx

API_KEY = "sk-..."  # Replace with your actual key
BASE_URL = "http://localhost:8004/v1"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "OpenAI-Beta": "assistants=v2",
}


def print_json(response):
    try:
        print(json.dumps(response.json(), indent=2))
    except Exception:
        print(response.text)


def post(url, payload=None):
    try:
        response = httpx.post(
            url,
            json=payload or {},
            headers=HEADERS,
            timeout=httpx.Timeout(60.0),
        )
        print_json(response)
        return response
    except Exception as e:
        print(f"[POST Error] {e}")
        return None


def get(url):
    try:
        response = httpx.get(url, headers=HEADERS, timeout=httpx.Timeout(60.0))
        print_json(response)
        return response
    except Exception as e:
        print(f"[GET Error] {e}")
        return None


def create_assistant():
    payload = {
        "name": "Historian",
        "description": "Summarizes historical topics.",
        "instructions": "You are a historian that summarizes documents.",
        "model": "Qwen--Qwen2.5-1.5B-Instruct",
        "tools": [],
        "file_ids": [],
        "metadata": {},
    }
    response = post(f"{BASE_URL}/assistants", payload)
    if response and response.status_code == 200:
        return response.json().get("id")


def create_thread():
    payload = {"messages": [], "metadata": {}}
    response = post(f"{BASE_URL}/threads", payload)
    if response and response.status_code == 200:
        return response.json().get("id")


def add_message(thread_id):
    payload = {
        "role": "user",
        "content": "Peux-tu résumer la Révolution française ?",
        "file_ids": [],
        "metadata": {},
    }
    post(f"{BASE_URL}/threads/{thread_id}/messages", payload)


def create_run(thread_id, assistant_id):
    payload = {
        "assistant_id": assistant_id,
        "model": None,
        "instructions": None,
        "additional_instructions": None,
        "additional_messages": None,
        "tools": None,
        "metadata": {},
        "temperature": None,
        "top_p": None,
        "max_prompt_tokens": None,
        "max_completion_tokens": None,
        "truncation_strategy": None,
        "tool_choice": None,
        "parallel_tool_calls": None,
        "response_format": None,
        "stream": False,
    }
    response = post(f"{BASE_URL}/threads/{thread_id}/runs", payload)
    if response and response.status_code == 200:
        return response.json().get("id")


def poll_run(thread_id, run_id):
    while True:
        response = get(f"{BASE_URL}/threads/{thread_id}/runs/{run_id}")
        if not response:
            return None
        try:
            run = response.json()
            status = run["status"]
        except Exception:
            return None
        print(f"[Run Status] {status}")
        if status in ["completed", "failed", "cancelled", "expired"]:
            return status
        if status == "requires_action":
            print("[Run requires tool outputs]")
            print_json(run)
            return status
        time.sleep(1)


def get_response(thread_id):
    response = get(f"{BASE_URL}/threads/{thread_id}/messages")
    if response and response.status_code == 200:
        messages = response.json().get("data", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if content and isinstance(content[0], dict):
                    value = content[0].get("text", {}).get("value")
                    if value:
                        print(f"\n[Assistant Response] {value}")
                        break


if __name__ == "__main__":
    assistant_id = create_assistant()
    if not assistant_id:
        exit(1)

    thread_id = create_thread()
    if not thread_id:
        exit(1)

    add_message(thread_id)
    run_id = create_run(thread_id, assistant_id)
    if not run_id:
        exit(1)

    status = poll_run(thread_id, run_id)
    if status == "completed":
        get_response(thread_id)
