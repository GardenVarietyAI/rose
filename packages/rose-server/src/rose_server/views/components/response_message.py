from datetime import datetime

from htpy import BaseElement, a, div, span


def response_message(
    *,
    uuid: str,
    dom_id: str | None,
    role: str,
    model: str | None,
    content: str,
    created_at: datetime | str | int,
    accepted: bool,
) -> BaseElement:
    model_text = f" | {model}" if model else ""

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

    return div(
        attrs,
        class_=("message accepted" if accepted else "message"),
        data_uuid=uuid,
        x_data="responseMessage",
    )[
        div(class_="message-header")[*header_children],
        div(class_="message-content")[content],
        div(class_="message-meta")[str(created_at)],
    ]
