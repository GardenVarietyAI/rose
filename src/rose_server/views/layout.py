from htpy import Node, a, body, h1, head, html, link, main, meta, nav, script, style, title


def render_page(*, title_text: str, content: Node) -> Node:
    return html(lang="en")[
        head[
            meta(charset="utf-8"),
            title[title_text],
            meta(name="viewport", content="width=device-width, initial-scale=1"),
            meta(name="color-scheme", content="light dark"),
            style(rel="text/css")["[x-cloak] { display: none !important }"],
            link(rel="stylesheet", href="/static/vendor/open-props/open-props.css"),
            link(rel="stylesheet", href="/static/app/app.css"),
        ],
        body[
            h1[a(href="/v1/search")["ROSE"]],
            nav(class_="navbar")[
                a(href="/v1/search")["Search"],
                # a(href="/v1/threads")["Threads"],
                a(href="/v1/lenses")["Lenses"],
            ],
            main(class_="container")[content],
            script(src="/static/app/bundle.js", defer=True),
            script(src="/static/vendor/alpine/alpine.min.js", defer=True),
        ],
    ]
