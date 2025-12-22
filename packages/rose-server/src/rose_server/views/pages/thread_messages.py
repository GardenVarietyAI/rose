from htpy import Node, div, span, template
from rose_server.models.messages import Message
from rose_server.views.components.response_message import response_message
from rose_server.views.pages.thread import render_thread_page


def render_thread_messages(*, thread_id: str, prompt: Message | None, responses: list[Message]) -> Node:
    content: list[Node] = []
    if prompt:
        content.append(
            div(class_="message", id=f"msg-{prompt.uuid}")[
                div(class_="message-header")[span(class_="message-role")[prompt.role]],
                div(class_="message-content")[prompt.content or ""],
                div(class_="message-meta")[str(prompt.created_at)],
            ]
        )

    response_nodes = []
    for resp in responses:
        response_nodes.append(
            response_message(
                uuid=resp.uuid,
                dom_id=f"msg-{resp.uuid}",
                role=resp.role,
                model=resp.model or "",
                content=resp.content or "",
                created_at=resp.created_at,
                accepted=resp.accepted_at is not None,
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

    return render_thread_page(
        thread_id=thread_id,
        prompt=prompt,
        active_tab="answers",
        content=div()[*content],
    )
