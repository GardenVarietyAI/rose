from htpy import Node, div

from rose_server.views.components.tabs import Tab, render_tabs
from rose_server.views.layout import render_page


def render_thread_page(
    *,
    thread_id: str,
    active_tab: str,
    content: Node,
) -> Node:
    tabs = [
        Tab(label="Answers", href=f"/v1/threads/{thread_id}", active=(active_tab == "answers")),
        Tab(label="Activity", href=f"/v1/threads/{thread_id}/activity", active=(active_tab == "activity")),
    ]

    return render_page(
        title_text=f"Thread: {thread_id}",
        content=div(x_data="threadPage()", data_thread_id=thread_id)[render_tabs(tabs=tabs), content],
    )
