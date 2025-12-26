from typing import Any

from htpy import Node, a, div


def render_search_result(*, hit: Any) -> Node:
    metadata = getattr(hit, "metadata", {}) or {}
    thread_id = metadata.get("thread_id")
    hit_id = getattr(hit, "id", "")
    score = getattr(hit, "score", "")
    excerpt = getattr(hit, "excerpt", "")

    return a(href=f"/v1/threads/{thread_id}#msg-{hit_id}")[
        div(class_="result")[
            div(class_="score")[f"Score: {float(score):.4f}" if score != "" else "Score:"],
            div(class_="content")[excerpt],
            div(class_="metadata")[
                f"Thread: {metadata.get('thread_id', '')} | ",
                f"Role: {metadata.get('role', '')} | ",
                f"Created: {metadata.get('created_at', '')}",
            ],
        ]
    ]
