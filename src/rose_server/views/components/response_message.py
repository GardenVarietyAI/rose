from datetime import datetime

from htpy import BaseElement, Node, a, div, span, template, textarea

from rose_server.views.components.message_card import message_card
from rose_server.views.components.time import render_time


def _snip_content(content: str, snip_chars: int) -> tuple[str, bool]:
    if len(content) <= snip_chars:
        return content, False

    if "```" in content:
        return content, False

    window = content[:snip_chars]
    paragraph_break = window.rfind("\n\n")
    if paragraph_break >= 0 and paragraph_break >= int(snip_chars * 0.3):
        cut = paragraph_break
    else:
        whitespace = window.rfind(" ")
        cut = whitespace if whitespace >= 0 else snip_chars

    snip = content[:cut].rstrip()
    return snip, True


def response_message(
    *,
    uuid: str,
    dom_id: str | None,
    role: str,
    display_role: str | None = None,
    model: str | None,
    content: str,
    created_at: datetime | str | int,
    accepted: bool,
    snip_chars: int | None = None,
) -> BaseElement:
    model_text = f" | {model}" if model else ""
    snipped_content, is_snipped = _snip_content(content, snip_chars) if snip_chars else (content, False)

    attrs: dict[str, str] = {":class": "{ accepted }"}

    header_children: list[Node] = [
        span(class_="message-role")[display_role or role],
        span(class_="message-model")[model_text],
    ]
    if role == "assistant":
        accept_label = "accepted" if accepted else "accept answer"
        header_children.extend(
            [
                " | ",
                a(
                    {":class": "{ accepted }", "@click.prevent": "toggleAccepted()"},
                    href="#",
                    class_="accept-link",
                    x_text="accepted ? 'accepted' : 'accept answer'",
                )[accept_label],
            ]
        )

    if role == "user":
        header_children.extend(
            [
                " | ",
                a({"x-show": "!editing", "@click.prevent": "startEdit()", "x-cloak": ""}, href="#")["edit"],
                span({"x-show": "editing", "x-cloak": ""})[
                    a({"@click.prevent": "cancelEdit()"}, href="#")["cancel"],
                    " | ",
                    a({"@click.prevent": "saveEdit()"}, href="#")["save"],
                ],
            ]
        )

    header_children.extend(
        [
            " | ",
            a(
                {
                    "@click.prevent": "collapsed = !collapsed",
                    "x-text": "collapsed ? 'expand' : 'collapse'",
                    "x-cloak": "",
                },
                href="#",
            )["collapse"],
        ]
    )

    content_node: BaseElement
    if is_snipped:
        content_node = div(class_="message-content")[
            div({"x-show": "!expanded", "x-cloak": ""})[
                snipped_content, "… ", a({"@click.prevent": "expanded = true"}, href="#")["Read more…"]
            ],
            div({"x-show": "expanded", "x-cloak": ""})[
                content, " ", a({"@click.prevent": "expanded = false"}, href="#")["Show less"]
            ],
        ]
    else:
        content_node = div(class_="message-content")[content]

    if role == "user":
        content_node = div()[
            template(x_ref="sourceContent")[content],
            div({"x-show": "!editing", "x-cloak": ""})[content_node],
            div({"x-show": "editing", "x-cloak": ""})[
                textarea(
                    {
                        "x-ref": "editor",
                        "x-model": "draft",
                        "@keydown.ctrl.enter.prevent": "saveEdit()",
                        "@keydown.meta.enter.prevent": "saveEdit()",
                    },
                    class_="message-editor",
                    rows="6",
                )
            ],
        ]

    return message_card(
        uuid=uuid,
        dom_id=dom_id,
        accepted=accepted,
        root_attrs=attrs,
        header_children=header_children,
        body_attrs={
            "x-show": "!collapsed",
            "x-cloak": "",
        },
        body_children=[
            content_node,
            div(class_="message-meta")[render_time(created_at)],
        ],
        x_data="responseMessage",
    )
