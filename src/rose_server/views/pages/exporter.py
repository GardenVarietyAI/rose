from htpy import (
    Node,
    button,
    div,
    form,
    input as input_,
    label,
    option,
    p,
    select,
    span,
)

from rose_server.models.message_types import LensMessage
from rose_server.models.messages import Message
from rose_server.views.app_data import AppData
from rose_server.views.layout import render_page


def render_export_page(lenses: list[Message]) -> Node:
    lens_options = []
    for lens_msg in lenses:
        try:
            lens = LensMessage(message=lens_msg)
            lens_options.append((lens.lens_id, lens.label))
        except Exception:
            continue

    content = div(class_="export-page", x_data="exportPage()")[
        form({"@submit.prevent": "generateExport()", "class": "export-form"})[
            div(class_="export-filters")[
                div(class_="filter-group")[
                    label[
                        input_(
                            {
                                "type": "checkbox",
                                "x-model": "acceptedOnly",
                            }
                        ),
                        " Only accepted responses",
                    ],
                ],
                div(class_="filter-group")[
                    label(for_="lens-select")["System message (lens):"],
                    select(
                        {
                            "id": "lens-select",
                            "x-model": "lensId",
                        }
                    )[
                        option(value="")["None (user-assistant only)"],
                        *[option(value=lens_id)[lens_label] for lens_id, lens_label in lens_options],
                    ],
                ],
                div(class_="filter-group")[
                    label(for_="split-ratio")["Train/valid split:"],
                    div(class_="split-control")[
                        input_(
                            {
                                "id": "split-ratio",
                                "type": "range",
                                "min": "0.5",
                                "max": "0.99",
                                "step": "0.01",
                                "x-model.number": "splitRatio",
                            }
                        ),
                        span(
                            x_text=(
                                "`${(splitRatio * 100).toFixed(0)}% train / "
                                "${((1 - splitRatio) * 100).toFixed(0)}% valid`"
                            )
                        ),
                    ],
                ],
            ],
            button(
                {
                    "type": "submit",
                    ":disabled": "generating",
                    "class": "btn-primary",
                }
            )[
                span(x_show="!generating", x_cloak="")["Generate Export"],
                span(x_show="generating", x_cloak="")["Generating..."],
            ],
        ],
        div(class_="export-results", x_show="exportId && stats", x_cloak="")[
            div(class_="export-stats")[
                p[
                    "Total conversations: ",
                    span(x_text="stats ? stats.total_conversations : 0"),
                ],
                p[
                    "Training set: ",
                    span(x_text="stats ? stats.train_count : 0"),
                    " conversations",
                ],
                p[
                    "Validation set: ",
                    span(x_text="stats ? stats.valid_count : 0"),
                    " conversations",
                ],
            ],
            div(class_="export-downloads")[
                button(
                    {
                        "@click": "downloadTrain()",
                        "type": "button",
                        "class": "btn-download",
                    }
                )["Download train.jsonl"],
                button(
                    {
                        "@click": "downloadValid()",
                        "type": "button",
                        "class": "btn-download",
                    }
                )["Download valid.jsonl"],
            ],
        ],
    ]

    return render_page(title_text="Export - ROSE", app_data=AppData(), content=content)
