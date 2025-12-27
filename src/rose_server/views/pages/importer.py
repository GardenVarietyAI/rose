from htpy import (
    Node,
    button,
    div,
    input as input_,
    label,
    option,
    p,
    select,
    span,
    table,
    tbody,
    td,
    template,
    th,
    thead,
    tr,
)

from rose_server.views.app_data import AppData
from rose_server.views.layout import render_page


def render_import() -> Node:
    content = div(class_="import-page", x_data="importPage()")[
        div(
            {
                "class": "import-banner",
                "x-show": "banner.visible",
                "x-cloak": "",
                ":class": "banner.type === 'error' ? 'import-banner-error' : 'import-banner-success'",
            }
        )[
            p(x_text="banner.message"),
            button({"@click": "closeBanner()", "type": "button", "class": "banner-close"})["×"],
        ],
        div(class_="import-section")[
            div(class_="import-format-selector")[
                label(for_="format-select")["Import format:"],
                select(
                    {
                        "id": "format-select",
                        "x-model": "selectedFormat",
                        "@change": "handleFormatChange()",
                    }
                )[option(value="claude-code")["Claude Code JSONL"]],
            ],
            input_(
                {
                    "type": "file",
                    "x-ref": "fileInput",
                    "@change": "handleFileSelect($event)",
                    ":accept": "currentValidator?.fileExtension || '.jsonl'",
                }
            ),
        ],
        div(class_="import-preview", x_show="preview", x_cloak="")[
            div(class_="import-stats")[
                p[
                    span(x_text="selectedCount")["0"],
                    " of ",
                    span(x_text="threads.length")["0"],
                    " threads selected (",
                    span(x_text="totalMessageCount")["0"],
                    " messages)",
                ],
                p(
                    {
                        "x-show": "parseReport && parseReport.skipped > 0",
                        "x-cloak": "",
                        "x-text": "parseReportSummary",
                        "class": "parse-report-warning",
                    }
                ),
            ],
            div(class_="import-table-wrapper")[
                table(class_="import-table")[
                    thead[
                        tr[
                            th[
                                input_(
                                    {
                                        "type": "checkbox",
                                        "@change": "toggleAll()",
                                        ":checked": "threads.every(t => t.selected)",
                                    }
                                )
                            ],
                            th["Question"],
                            th["Responses"],
                        ]
                    ],
                    tbody[
                        template({"x-for": "thread in threads", ":key": "thread.id"})[
                            tr[
                                td[input_({"type": "checkbox", "x-model": "thread.selected"})],
                                td(x_text="thread.userMessage.content.substring(0, 200)"),
                                td(x_text="thread.assistantMessages.length"),
                            ]
                        ]
                    ],
                ]
            ],
            div(class_="import-actions")[
                button({"@click": "cancelImport()"}, type="button")["Cancel"],
                button({"@click": "executeImport()", ":disabled": "importing || selectedCount === 0"}, type="button")[
                    span(x_show="!importing", x_cloak="")["Import Selected"],
                    span(x_show="importing", x_cloak="")["Importing..."],
                ],
            ],
        ],
        div(class_="import-complete", x_show="complete", x_cloak="")[
            p["Import complete!"],
            button({"@click": "window.location.href = '/v1/search'"}, type="button")["Go to Search"],
        ],
    ]

    return render_page(title_text="Import - ROSE", app_data=AppData(), content=content)
