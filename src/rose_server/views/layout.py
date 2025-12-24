from htpy import Node, a, aside, body, div, h1, head, header, html, link, main, meta, nav, script, style, title


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
            header(class_="mobile-header")[
                h1[a(href="/v1/search")["ROSE"]],
                a(href="/v1/nav", class_="menu-link")["Menu"],
            ],
            div(class_="app-layout")[
                nav(class_="nav-sidebar")[
                    h1[a(href="/v1/search")["ROSE"]],
                    a(href="/v1/search")["Search"],
                    a(href="/v1/lenses")["Lenses"],
                ],
                main(class_="main-content")[content],
                aside(class_="aside")[div(class_="aside-placeholder")["Filters & context"],],
            ],
            script(src="/static/app/bundle.js", defer=True),
            script(src="/static/vendor/alpine/alpine.min.js", defer=True),
        ],
    ]
