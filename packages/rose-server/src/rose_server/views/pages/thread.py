from typing import Any

from htpy import Node, a, div, p, script, span, strong, template
from rose_server.views.components.response_message import response_message
from rose_server.views.layout import render_page


def render_thread(*, thread_id: str, prompt: Any, responses: list[Any]) -> Node:
    content = []

    content.append(
        p[
            "Thread ID: ",
            strong[thread_id],
            " | ",
            a(
                {
                    "@click.prevent": "regenerate($event)",
                },
                href="#",
                class_="regenerate-link",
                data_thread_id=thread_id,
                data_prompt_content=getattr(prompt, "content", "") if prompt else "",
                data_model=getattr(prompt, "model", "") if prompt else "",
            )["ask again"],
        ]
    )

    if prompt:
        content.append(
            div(class_="message", id=f"msg-{getattr(prompt, 'uuid', '')}")[
                div(class_="message-header")[span(class_="message-role")[getattr(prompt, "role", "")]],
                div(class_="message-content")[getattr(prompt, "content", "")],
                div(class_="message-meta")[str(getattr(prompt, "created_at", ""))],
            ]
        )

    response_nodes = []
    for resp in responses:
        response_nodes.append(
            response_message(
                uuid=getattr(resp, "uuid", ""),
                dom_id=f"msg-{getattr(resp, 'uuid', '')}",
                role=getattr(resp, "role", ""),
                model=getattr(resp, "model", "") or "",
                content=getattr(resp, "content", ""),
                created_at=getattr(resp, "created_at", ""),
                accepted=bool(getattr(resp, "accepted_at", None)),
            )
        )
    content.append(div(id="responses", x_ref="responses")[*response_nodes])

    content.append(
        template(id="placeholder-template", x_ref="placeholderTemplate")[
            div(class_="message placeholder", data_temp_id="")[
                div(class_="message-header")[
                    span(class_="message-role")["assistant"],
                    span(class_="message-model")[""],
                ],
                div(class_="message-content")[div(class_="spinner")[""]],
                div(class_="message-meta")[""],
            ]
        ]
    )

    content.append(script(src="/static/app/pages/thread.js", defer=True))

    return render_page(title_text=f"Thread: {thread_id}", content=div(x_data="threadPage")[content])
