from htpy import (
    Node,
    div,
    span,
)

from rose_server.models.job_events import JobEvent
from rose_server.views.components.message_card import message_card
from rose_server.views.components.time import render_time
from rose_server.views.pages.thread import render_thread_page


def render_thread_activity(
    *,
    thread_id: str,
    job_events: list[JobEvent],
) -> Node:
    items: list[Node] = []
    for job_event in job_events:
        summary_parts: list[str] = [job_event.status]
        if job_event.attempt > 0:
            summary_parts.append(f"attempt {job_event.attempt}")
        if job_event.error:
            summary_parts.append(job_event.error)

        summary = " | ".join(summary_parts)
        items.append(
            message_card(
                uuid=job_event.uuid,
                dom_id=None,
                header_children=[
                    span(class_="message-role")["job"],
                    span(class_="message-model")[f" | {job_event.job_id}"],
                ],
                body_children=[
                    div(class_="message-content")[summary],
                    div(class_="message-meta")[render_time(job_event.created_at)],
                ],
            )
        )

    return render_thread_page(
        thread_id=thread_id,
        active_tab="activity",
        content=div()[*items],
        x_data=None,
    )
