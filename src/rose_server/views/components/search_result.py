from typing import Any

from htpy import Node, a, div, span, strong


def render_search_result(*, hit: Any) -> Node:
    thread_id = hit.thread_id
    score = hit.score
    user_text = hit.user_message_text
    user_excerpt = getattr(hit, "user_message_excerpt", None)
    assistant_excerpt = hit.assistant_message_excerpt
    assistant_model = hit.assistant_message_model
    accepted = hit.accepted
    matched_message_id = hit.matched_message_id

    user_preview = user_excerpt if user_excerpt else (user_text[:150] + "..." if len(user_text) > 150 else user_text)

    return a(href=f"/v1/threads/{thread_id}#msg-{matched_message_id}")[
        div(class_="thread-result")[
            div(class_="score")[f"Score: {float(score):.4f}"],
            div(class_="conversation")[
                div(class_="question")[
                    strong["Q: "],
                    span[user_preview],
                ],
                div(class_="answer")[
                    strong["A: "],
                    span[assistant_excerpt],
                    span(class_="accepted")[" (accepted)"] if accepted else "",
                ],
            ],
            div(class_="metadata")[
                f"Thread: {thread_id}",
                f" | Model: {assistant_model}" if assistant_model else "",
            ],
        ]
    ]
