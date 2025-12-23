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


def render_lenses_page(*, lenses: list[Message]) -> Node:
    content: list[Node] = [
        render_tabs(
            tabs=[
                Tab(label="Lenses", href="/v1/lenses", active=True),
                Tab(label="Create", href="/v1/lenses/create", active=False),
            ]
        ),
        h2["Lenses"],
    ]

    for lens in lenses:
        meta = lens.meta
        if meta is None:
            raise RuntimeError("Lens missing meta")
        if lens.content is None:
            raise RuntimeError("Lens missing content")
        label_text = meta["label"]
        at_name = meta["at_name"]
        system_prompt = lens.content
        content.append(
            a(href=f"/v1/lenses/{lens.uuid}/edit")[
                div(class_="result")[
                    div(class_="content")[label_text],
                    div(class_="metadata")[f"@{at_name}"],
                    div(class_="content")[_snip(system_prompt)],
                ]
            ]
        )

    return render_page(title_text="Lenses", content=div()[*content])


def render_lens_form_page(*, lens: Message | None) -> Node:
    is_edit = lens is not None
    second_label = "Edit" if is_edit else "Create"
    title = f"{second_label} lens"

    if lens is None:
        lens_id = ""
        at_name = ""
        label_value = ""
        system_prompt = ""
        action = "/v1/lenses"
    else:
        meta = lens.meta
        if meta is None:
            raise RuntimeError("Lens missing meta")
        if lens.content is None:
            raise RuntimeError("Lens missing content")
        lens_id = lens.uuid
        at_name = meta["at_name"]
        label_value = meta["label"]
        system_prompt = lens.content
        action = f"/v1/lenses/{lens.uuid}"

    content: list[Node] = [
        render_tabs(
            tabs=[
                Tab(label="Lenses", href="/v1/lenses", active=False),
                Tab(
                    label=second_label,
                    href=f"/v1/lenses/{lens_id}/edit" if is_edit else "/v1/lenses/create",
                    active=True,
                ),
            ]
        ),
        h2[title],
        div[
            form(action=action, method="post", class_="lens-form")[
                label["@name"],
                input_(
                    type="text",
                    name="at_name",
                    value=at_name,
                    placeholder="e.g. softwareengineer",
                    pattern="[A-Za-z0-9]+",
                ),
                label["Label"],
                input_(type="text", name="label", value=label_value),
                label["System prompt"],
                textarea(name="system_prompt", rows="12")[system_prompt],
                input_(type="submit", value="Save"),
            ],
            form(action=f"/v1/lenses/{lens_id}/delete", method="post")[
                input_(type="submit", value="Delete", disabled=(not is_edit)),
            ],
        ],
    ]

    return render_page(title_text=title, content=div()[*content])
