from htpy import (
    Node,
    button,
    div,
    form,
    input as input_,
    label,
    p,
    span,
)

from rose_server.views.app_data import AppData
from rose_server.views.layout import render_page


def render_import_page() -> Node:
    content = div(class_="import-page", x_data="importPage()")[
        form(
            {
                "@submit.prevent": "submitImport()",
                "class": "import-form",
            }
        )[
            div(class_="import-fields")[
                div(class_="field-group")[
                    label(for_="file-input")["JSONL file:"],
                    input_(
                        {
                            "id": "file-input",
                            "type": "file",
                            "accept": ".jsonl",
                            "@change": "handleFile($event)",
                            "required": True,
                        }
                    ),
                ],
                div(class_="field-group")[
                    label(for_="import-source")["Import source:"],
                    input_(
                        {
                            "id": "import-source",
                            "type": "text",
                            "x-model": "importSource",
                            "placeholder": "e.g., training-data-2024",
                            "required": True,
                        }
                    ),
                ],
            ],
            div(class_="import-preview", x_show="conversations.length > 0", x_cloak="")[
                p[
                    "Found ",
                    span(x_text="conversations.length"),
                    " conversations with ",
                    span(x_text="totalMessages"),
                    " total messages",
                ],
            ],
            button(
                {
                    "type": "submit",
                    ":disabled": "importing || conversations.length === 0",
                    "class": "btn-primary",
                }
            )[
                span(x_show="!importing", x_cloak="")["Import"],
                span(x_show="importing", x_cloak="")["Importing..."],
            ],
        ],
        div(class_="import-results", x_show="stats", x_cloak="")[
            p["Import complete!"],
            p[
                "Imported ",
                span(x_text="stats?.imported_count"),
                " messages in ",
                span(x_text="stats?.conversations_count"),
                " conversations",
            ],
            button(
                {
                    "@click": "window.location.href = '/v1/threads'",
                    "type": "button",
                }
            )["View Threads"],
        ],
    ]

    return render_page(title_text="Import - ROSE", app_data=AppData(), content=content)
