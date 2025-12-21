from typing import Any, Iterable, Optional
from urllib.parse import quote

from htpy import (
    Node,
    a,
    br,
    div,
    form,
    input as input_,
    p,
    span,
    strong,
)
from rose_server.views.components.ask_button import ask_button
from rose_server.views.layout import render_page


def render_search(
    *,
    query: str,
    hits: Iterable[Any],
    corrected_query: Optional[str],
    original_query: str,
    fallback_keywords: Optional[list[str]],
) -> Node:
    content: list[Node] = []
    content.append(
        form(action="/v1/search", method="get")[
            input_(type="text", name="q", value=query, autofocus=True),
            input_(type="submit", value="Search"),
            ask_button(),
        ]
    )

    if corrected_query:
        content.append(
            p[
                "Showing results for ",
                strong[query],
                br(),
                "Search for ",
                a(href=f"/v1/search?q={quote(original_query)}&exact=true")[
                    span(class_="original-query")[original_query]
                ],
            ]
        )

    hits_list = list(hits)
    if fallback_keywords and len(hits_list) == 0:
        links: list[Node] = []
        for i, keyword in enumerate(fallback_keywords):
            links.append(a(href=f"/v1/search?q={quote(keyword)}")[span(class_="original-query")[keyword]])
            if i != len(fallback_keywords) - 1:
                links.append(", ")
        content.append(p["No matches. Try searching for: ", *links])
    else:
        content.append(p[f"Results: {len(hits_list)}"])

    for hit in hits_list:
        metadata = getattr(hit, "metadata", {}) or {}
        thread_id = metadata.get("thread_id")
        hit_id = getattr(hit, "id", "")
        score = getattr(hit, "score", "")
        excerpt = getattr(hit, "excerpt", "")
        content.append(
            a(href=f"/v1/threads/{thread_id}#msg-{hit_id}")[
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
        )

    return render_page(title_text=f"Search: {query}", content=content)
