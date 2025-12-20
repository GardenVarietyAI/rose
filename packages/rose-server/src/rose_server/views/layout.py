from htpy import Node, a, body, h1, head, html, link, main, meta, script, title


def render_page(*, title_text: str, content: Node) -> Node:
    return html(lang="en")[
        head[
            meta(charset="utf-8"),
            title[title_text],
            meta(name="viewport", content="width=device-width, initial-scale=1"),
            meta(name="color-scheme", content="light"),
            link(rel="stylesheet", href="/static/vendor/sanitize/sanitize.css"),
            link(rel="stylesheet", href="/static/app/app.css"),
        ],
        body[
            h1[a(href="/v1/search")["ROSE"]],
            main(class_="container")[content],
            script(src="/static/app/app.js", defer=True),
            script(src="/static/app/components/ask-button.js", defer=True),
            script(src="/static/app/components/response-message.js", defer=True),
            script(src="/static/vendor/alpine/alpine.min.js", defer=True),
        ],
    ]
