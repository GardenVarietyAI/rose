from htpy import (
    Node,
    a,
    div,
    form,
    h2,
    input as input_,
    label,
    textarea,
)

from rose_server.models.messages import Message
from rose_server.views.components.tabs import Tab, render_tabs
from rose_server.views.layout import render_page


def _snip(text: str, max_chars: int = 180) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def render_factsheets_page(*, factsheets: list[Message]) -> Node:
    content: list[Node] = [
        render_tabs(
            tabs=[
                Tab(label="Factsheets", href="/v1/factsheets", active=True),
                Tab(label="Create", href="/v1/factsheets/create", active=False),
            ]
        ),
        h2["Factsheets"],
    ]

    for factsheet in factsheets:
        meta = factsheet.meta
        if meta is None:
            raise RuntimeError("Fact sheet missing meta")
        if factsheet.content is None:
            raise RuntimeError("Fact sheet missing content")
        title_text = meta["title"]
        tag = meta["tag"]
        body = factsheet.content
        content.append(
            a(href=f"/v1/factsheets/{factsheet.uuid}/edit")[
                div(class_="result")[
                    div(class_="content")[title_text],
                    div(class_="metadata")[f"#{tag}"],
                    div(class_="content")[_snip(body)],
                ]
            ]
        )

    return render_page(title_text="Factsheets", content=div()[*content])


def render_factsheet_form_page(*, factsheet: Message | None) -> Node:
    is_edit = factsheet is not None
    second_label = "Edit" if is_edit else "Create"
    title = f"{second_label} factsheet"

    if factsheet is None:
        factsheet_id = ""
        at_name = ""
        title_value = ""
        body_value = ""
        action = "/v1/factsheets"
    else:
        meta = factsheet.meta
        if meta is None:
            raise RuntimeError("Fact sheet missing meta")
        if factsheet.content is None:
            raise RuntimeError("Fact sheet missing content")
        factsheet_id = factsheet.uuid
        at_name = meta["tag"]
        title_value = meta["title"]
        body_value = factsheet.content
        action = f"/v1/factsheets/{factsheet.uuid}"

    content: list[Node] = [
        render_tabs(
            tabs=[
                Tab(label="Factsheets", href="/v1/factsheets", active=False),
                Tab(
                    label=second_label,
                    href=f"/v1/factsheets/{factsheet_id}/edit" if is_edit else "/v1/factsheets/create",
                    active=True,
                ),
            ]
        ),
        h2[title],
        div[
            form(action=action, method="post", class_="lens-form")[
                label["#tag"],
                input_(
                    type="text",
                    name="tag",
                    value=at_name,
                    placeholder="e.g. productfaq",
                    pattern="[A-Za-z0-9]+",
                ),
                label["Title"],
                input_(type="text", name="title", value=title_value),
                label["Body"],
                textarea(name="body", rows="12")[body_value],
                input_(type="submit", value="Save"),
            ],
            form(action=f"/v1/factsheets/{factsheet_id}/delete", method="post")[
                input_(type="submit", value="Delete", disabled=(not is_edit)),
            ],
        ],
    ]

    return render_page(title_text=title, content=div()[*content])
