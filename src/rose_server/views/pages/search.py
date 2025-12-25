from typing import Any, Iterable
from urllib.parse import quote

from htpy import (
    Node,
    a,
    br,
    button,
    div,
    form,
    input as input_,
    option,
    p,
    select,
    span,
    strong,
    textarea,
)

from rose_server.views.components.ask_button import ask_button
from rose_server.views.layout import render_page


def render_search(
    *,
    query: str,
    hits: Iterable[Any],
    corrected_query: str | None,
    original_query: str,
    lenses: list[tuple[str, str]],
    selected_lens_id: str | None = None,
) -> Node:
    hits_list = list(hits)
    content: list[Node] = []
    content.append(
        form(
            {
                "x-ref": "form",
                "data-initial-query": query,
                "data-initial-lens-id": selected_lens_id or "",
                "@submit.prevent": "submit()",
            },
            action="/v1/search",
            method="get",
            autocomplete="off",
            class_="search-form",
            x_data="searchForm",
        )[
            input_(
                {
                    "x-ref": "single",
                    "x-show": "!isMultiline",
                    "x-cloak": "",
                    "x-model": "value",
                    ":disabled": "isMultiline",
                    "@paste": "handlePaste($event)",
                    "@keydown.enter.prevent": "submit()",
                },
                type="text",
                name="q",
                autofocus=True,
            ),
            textarea(
                {
                    "x-ref": "multi",
                    "x-show": "isMultiline",
                    "x-cloak": "",
                    "x-model": "value",
                    ":disabled": "!isMultiline",
                    "@input": "autogrow($event.target)",
                    "@keydown.enter.prevent": "submit()",
                },
                name="q",
                rows="1",
            )[query],
            div(class_="search-controls-row")[
                span(class_="search-results-count")[f"Results: {len(hits_list)}"],
                div(class_="search-actions")[
                    button(
                        {
                            "@click.prevent": "settingsOpen = !settingsOpen",
                            "x-text": "settingsOpen ? 'Settings' : 'Settings'",
                        },
                        type="button",
                        class_="settings-button",
                    )["Settings"],
                    input_(type="submit", value="Search"),
                    ask_button(),
                ],
            ],
            div(
                {"x-show": "settingsOpen", "x-cloak": ""},
                class_="settings-panel",
            )[
                select(
                    {"x-ref": "lensSelect", "name": "lens_id", "x-model": "$store.search.lensId"},
                    class_="settings-select",
                )[
                    option({"value": "", **({"selected": ""} if not selected_lens_id else {})})["No lens"],
                    *[
                        option({"value": lens_id, **({"selected": ""} if lens_id == selected_lens_id else {})})[label]
                        for lens_id, label in lenses
                    ],
                ],
                div(class_="settings-actions")[
                    button({"@click.prevent": "clearLenses()"}, type="button", class_="settings-clear-button")["Clear"]
                ],
            ],
        ]
    )

    if corrected_query:
        content.append(
            p[
                "Searching for ",
                strong[corrected_query],
                br(),
                "Search for ",
                a(href=f"/v1/search?q={quote(original_query)}&exact=true")[
                    span(class_="original-query")[original_query]
                ],
            ]
        )

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

    return render_page(title_text=f"Search: {query}", content=div(id="search-root")[*content])
