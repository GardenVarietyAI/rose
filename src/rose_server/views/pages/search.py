from typing import Any, Iterable
from urllib.parse import quote

from htpy import Node, a, br, div, p, span, strong

from rose_server.views.app_data import AppData, SearchAppData
from rose_server.views.components.search_form import render_search_form
from rose_server.views.components.search_result import render_search_result
from rose_server.views.layout import render_page


def render_search(
    *,
    query: str,
    hits: Iterable[Any],
    corrected_query: str | None,
    original_query: str,
    lenses: list[tuple[str, str, str]],
    selected_lens_id: str | None = None,
    limit: int = 10,
    exact: bool = False,
) -> Node:
    hits_list = list(hits)
    content: list[Node] = []
    content.append(render_search_form(lenses=lenses, selected_lens_id=selected_lens_id, hits_count=len(hits_list)))

    if corrected_query:
        content.append(
            p[
                "Searching for ",
                strong[corrected_query],
                br(),
                "Search for ",
                a(href=f"/v1/search?q={quote(original_query)}&exact=true")[
                    span(class_="original-query")[original_query]
                ],
            ]
        )

    for hit in hits_list:
        content.append(render_search_result(hit=hit))

    return render_page(
        title_text=f"Search: {query}",
        app_data=AppData(
            search=SearchAppData(
                lens_map={at_name: lens_id for lens_id, at_name, _label in lenses if at_name},
                query=query,
                lens_id=selected_lens_id or "",
                limit=limit,
                exact=exact,
            )
        ),
        content=div(id="search-root")[*content],
    )
