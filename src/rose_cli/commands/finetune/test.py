from typing import Any

import typer

from ...utils import console, get_client


def test_fine_tuned_model(
    job_id: str = typer.Argument(..., help="Fine-tuning job ID"),
):
    """Test a fine-tuned model with sample prompts."""
    client = get_client()

    try:
        # Get the fine-tuning job
        job = client.fine_tuning.jobs.retrieve(job_id)

        if job.status != "succeeded":
            console.print(f"[red]Job {job_id} is not complete. Status: {job.status}[/red]")
            return

        if not job.fine_tuned_model:
            console.print(f"[red]No fine-tuned model found for job {job_id}[/red]")
            return

        console.print(f"[green]Testing model: {job.fine_tuned_model}[/green]\n")

        # Test prompts
        test_prompts = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello! What can you do?"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Write a Python function to calculate factorial."},
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the weather like today?",
                    },
                    {
                        "role": "assistant",
                        "content": "I don't have access to current weather data, "
                        "but you can check your local weather service!",
                    },
                    {"role": "user", "content": "Thanks! Can you help me with Python instead?"},
                ]
            },
        ]

        for i, prompt in enumerate(test_prompts, 1):
            console.print(f"[cyan]Test {i}:[/cyan]")
            for msg in prompt["messages"]:
                console.print(f"  [{msg['role']}]: {msg['content']}")

            messages: list[Any] = prompt["messages"]
            response = client.chat.completions.create(
                model=job.fine_tuned_model,
                messages=messages,
                max_tokens=150,
            )

            console.print(f"  [assistant]: {response.choices[0].message.content}\n")

    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
