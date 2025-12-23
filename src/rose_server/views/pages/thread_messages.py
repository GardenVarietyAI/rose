from htpy import Node, a, div, option, p, select, span, strong, template

from rose_server.models.messages import Message
from rose_server.views.components.response_message import response_message
from rose_server.views.pages.thread import render_thread_page

_PROMPT_SNIP_CHARS = 500


def render_thread_messages(
    *,
    thread_id: str,
    prompt: Message | None,
    responses: list[Message],
    lenses: list[tuple[str, str]],
    selected_lens_id: str | None = None,
) -> Node:
    header = p[
        strong["Lens: "],
        select({"x-ref": "lensSelect"}, class_="lens-select")[
            option(value="")["Default"],
            *[
                option({"value": lens_id, **({"selected": ""} if selected_lens_id == lens_id else {})})[label]
                for lens_id, label in lenses
            ],
        ],
        " | ",
        a(
            {"@click.prevent": "regenerate($event)"},
            href="#",
            class_="regenerate-link",
            data_thread_id=thread_id,
            data_model=(prompt.model or "") if prompt else "",
        )["Ask Again"],
    ]

    content: list[Node] = []
    if prompt:
        content.append(
            response_message(
                uuid=prompt.uuid,
                dom_id=f"msg-{prompt.uuid}",
                role=prompt.role,
                model=None,
                content=prompt.content or "",
                created_at=prompt.created_at,
                accepted=False,
                snip_chars=_PROMPT_SNIP_CHARS,
            )
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
                div(class_="message-body")[
                    div(class_="message-content")[div(class_="spinner")[""]],
                    div(class_="message-meta")[""],
                ],
            ]
        ]
    )

    return render_thread_page(
        thread_id=thread_id,
        active_tab="answers",
        content=div()[header, *content],
    )
