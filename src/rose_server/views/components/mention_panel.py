from htpy import Node, button, div, span, template


def render_mention_panel() -> Node:
    return div(
        {"x-show": "mentionOpen", "x-cloak": ""},
        class_="mention-panel",
    )[
        template({"x-for": "(option, index) in mentionOptions", "x-bind:key": "option.lensId || option.factsheetId"})[
            button(
                {
                    "type": "button",
                    "@click": "selectMention(option)",
                    "x-bind:class": "index === mentionIndex ? 'mention-option is-active' : 'mention-option'",
                }
            )[
                span({"x-text": "option.tag ? `#${option.tag}` : `@${option.atName}`"}, class_="mention-name"),
                span({"x-text": "option.title || option.label || ''"}, class_="mention-label"),
            ]
        ]
    ]
