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
    template,
)

from rose_server.views.app_data import AppData, SearchAppData
from rose_server.views.components.ask_button import ask_button
from rose_server.views.layout import render_page


def render_search(
    *,
    query: str,
    hits: Iterable[Any],
    corrected_query: str | None,
    original_query: str,
    lenses: list[tuple[str, str, str]],
    selected_lens_id: str | None = None,
    limit: int = 10,
    exact: bool = False,
) -> Node:
    hits_list = list(hits)
    content: list[Node] = []
    content.append(
        form(
            {
                "x-ref": "form",
                "@submit.prevent": "submit()",
            },
            action="/v1/search",
            method="get",
            autocomplete="off",
            class_="search-form",
            x_data="searchForm",
        )[
            input_(
                {"x-ref": "textarea", "x-model": "queryValue"},
                type="hidden",
                name="q",
            ),
            div(
                {
                    "x-ref": "editor",
                    "x-on:input": "syncFromEditor()",
                    "@keydown": "handleEditorKeydown($event)",
                    "contenteditable": "true",
                    "role": "textbox",
                    "tabindex": "0",
                    "aria-multiline": "true",
                    "spellcheck": "false",
                },
                class_="search-editor",
            ),
            div(
                {"x-show": "mentionOpen", "x-cloak": ""},
                class_="mention-panel",
            )[
                template({"x-for": "(option, index) in mentionOptions", "x-bind:key": "option.lensId"})[
                    button(
                        {
                            "type": "button",
                            "@click": "selectMention(option)",
                            "x-bind:class": "index === mentionIndex ? 'mention-option is-active' : 'mention-option'",
                        }
                    )[span({"x-text": "`@${option.atName}`"}, class_="mention-name"),]
                ]
            ],
            div(class_="search-controls-row")[
                span(class_="search-results-count")[f"Results: {len(hits_list)}"],
                div({"x-show": "submitting", "x-cloak": ""}, class_="search-status")[div(class_="spinner")[""]],
                div(class_="search-actions")[
                    button(
                        {
                            "@click.prevent": "settingsOpen = !settingsOpen",
                            "x-text": "settingsOpen ? 'Settings' : 'Settings'",
                        },
                        type="button",
                        class_="settings-button",
                    )["Settings"],
                    input_({":disabled": "submitting"}, type="submit", value="Search"),
                    ask_button(),
                ],
            ],
            div(
                {"x-show": "settingsOpen", "x-cloak": ""},
                class_="settings-panel",
            )[
                select(
                    {
                        "x-ref": "lensSelect",
                        "name": "lens_id",
                        "x-model": "$store.search.lens_id",
                        "@change": "syncLensToken()",
                    },
                    class_="settings-select",
                )[
                    option({"value": "", **({"selected": ""} if not selected_lens_id else {})})["No lens"],
                    *[
                        option(
                            {
                                "value": lens_id,
                                **({"selected": ""} if lens_id == selected_lens_id else {}),
                            }
                        )[label]
                        for lens_id, at_name, label in lenses
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

    return render_page(
        title_text=f"Search: {query}",
        app_data=AppData(
            search=SearchAppData(
                lens_map={at_name: lens_id for lens_id, at_name, _label in lenses if at_name},
                query=query,
                lens_id=selected_lens_id or "",
                limit=limit,
                exact=exact,
            )
        ),
        content=div(id="search-root")[*content],
    )
