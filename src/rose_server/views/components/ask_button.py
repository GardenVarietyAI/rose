from htpy import BaseElement, button


def ask_button() -> BaseElement:
    return button(
        {
            ":disabled": "disabled",
            "@click": "ask()",
        },
        type="button",
        class_="ask-btn",
        x_data="askButton()",
        x_text="label",
    )["Ask"]
