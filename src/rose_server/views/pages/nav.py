from htpy import Node, a, div, nav

from rose_server.views.layout import render_page


def render_nav() -> Node:
    return render_page(
        title_text="Navigation - ROSE",
        content=div(class_="nav-page")[
            nav(class_="nav-page-links")[
                a(href="/v1/search")["Search"],
                a(href="/v1/threads")["Threads"],
                a(href="/v1/factsheets")["Factsheets"],
                a(href="/v1/lenses")["Lenses"],
                a(href="/v1/import")["Import"],
            ],
        ],
    )
