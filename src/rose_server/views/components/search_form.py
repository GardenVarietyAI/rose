from htpy import (
    Node,
    button,
    div,
    form,
    input as input_,
    option,
    select,
    span,
)

from rose_server.views.components.ask_button import ask_button
from rose_server.views.components.mention_panel import render_mention_panel


def render_search_form(
    *,
    lenses: list[tuple[str, str, str]],
    selected_lens_id: str | None,
    hits_count: int,
) -> Node:
    return form(
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
        render_mention_panel(),
        div(class_="search-controls-row")[
            span(class_="search-results-count")[f"Results: {hits_count}"],
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
                    for lens_id, _at_name, label in lenses
                ],
            ],
            div(class_="settings-actions")[
                button({"@click.prevent": "clearLenses()"}, type="button", class_="settings-clear-button")["Clear"]
            ],
        ],
    ]
