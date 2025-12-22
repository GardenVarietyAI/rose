import json

from htpy import (
    Node,
    div,
    pre,
    span,
)
from rose_server.models.messages import Message
from rose_server.views.components.message_card import message_card
from rose_server.views.components.time import render_time
from rose_server.views.pages.thread import render_thread_page


def render_thread_activity(
    *,
    thread_id: str,
    prompt: Message | None,
    system_messages: list[Message],
) -> Node:
    items: list[Node] = []
    for message in system_messages:
        meta = message.meta
        summary_parts: list[str] = []
        meta_text = ""
        if meta:
            meta_text = json.dumps(meta, indent=2, sort_keys=True)
            status = meta.get("status")
            job_name = meta.get("job_name")
            error = meta.get("error")
            if status:
                summary_parts.append(str(status))
            if job_name:
                summary_parts.append(str(job_name))
            if error:
                summary_parts.append(str(error))

        summary = " | ".join(summary_parts)
        items.append(
            message_card(
                uuid=message.uuid,
                dom_id=None,
                header_children=[
                    span(class_="message-role")[message.role],
                    span(class_="message-model")[f" | {message.model}" if message.model else ""],
                ],
                body_children=[
                    div(class_="message-content")[message.content or ""],
                    div(class_="message-meta")[summary if summary else render_time(message.created_at)],
                    pre(class_="message-meta")[meta_text] if meta_text else "",
                ],
            )
        )

    return render_thread_page(
        thread_id=thread_id,
        active_tab="activity",
        content=div()[*items],
    )
