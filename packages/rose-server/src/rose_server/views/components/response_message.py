from datetime import datetime

from htpy import BaseElement, a, div, span
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
    model: str | None,
    content: str,
    created_at: datetime | str | int,
    accepted: bool,
    snip_chars: int | None = None,
) -> BaseElement:
    model_text = f" | {model}" if model else ""
    snipped_content, is_snipped = _snip_content(content, snip_chars) if snip_chars else (content, False)

    attrs: dict[str, str] = {
        ":class": "{ accepted }",
    }
    if dom_id:
        attrs["id"] = dom_id

    header_children: list[BaseElement | str] = [
        span(class_="message-role")[role],
        span(class_="message-model")[model_text],
    ]
    if role == "assistant":
        accept_label = "accepted" if accepted else "accept answer"
        header_children.extend(
            [
                " | ",
                a(
                    {
                        ":class": "{ accepted }",
                        "@click.prevent": "toggleAccepted()",
                    },
                    href="#",
                    class_="accept-link",
                    x_text="accepted ? 'accepted' : 'accept answer'",
                )[accept_label],
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
            div(
                {
                    "x-show": "!expanded",
                    "x-cloak": "",
                }
            )[
                snipped_content,
                "… ",
                a({"@click.prevent": "expanded = true"}, href="#")["Read more…"],
            ],
            div(
                {
                    "x-show": "expanded",
                    "x-cloak": "",
                }
            )[
                content,
                " ",
                a({"@click.prevent": "expanded = false"}, href="#")["Show less"],
            ],
        ]
    else:
        content_node = div(class_="message-content")[content]

    return div(
        attrs,
        class_=("message accepted" if accepted else "message"),
        data_uuid=uuid,
        x_data="responseMessage",
    )[
        div(class_="message-header")[*header_children],
        div(
            {
                "class": "message-body",
                "x-show": "!collapsed",
                "x-cloak": "",
            }
        )[
            content_node,
            div(class_="message-meta")[render_time(created_at)],
        ],
    ]
