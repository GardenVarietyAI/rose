from htpy import Node, a, div, p, strong
from rose_server.models.messages import Message
from rose_server.views.components.tabs import Tab, render_tabs
from rose_server.views.layout import render_page


def render_thread_page(
    *,
    thread_id: str,
    prompt: Message | None,
    active_tab: str,
    content: Node,
) -> Node:
    tabs = [
        Tab(label="Answers", href=f"/v1/threads/{thread_id}", active=(active_tab == "answers")),
        Tab(label="Activity", href=f"/v1/threads/{thread_id}/activity", active=(active_tab == "activity")),
    ]

    header = p[
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
            data_prompt_content=(prompt.content or "") if prompt else "",
            data_model=(prompt.model or "") if prompt else "",
        )["ask again"],
    ]

    return render_page(
        title_text=f"Thread: {thread_id}",
        content=div(x_data="threadPage", data_thread_id=thread_id)[header, render_tabs(tabs=tabs), content],
    )
