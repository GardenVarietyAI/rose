import base64
import json
from typing import Any, Iterable
from urllib.parse import quote

from htpy import Node, a, br, div, p, span, strong

from rose_server.views.app_data import AppData, SearchAppData
from rose_server.views.components.search_form import render_search_form
from rose_server.views.components.search_result import render_search_result
from rose_server.views.layout import render_page


def render_search_root(
    *,
    query: str,
    hits: Iterable[Any],
    corrected_query: str | None,
    original_query: str,
    lenses: list[tuple[str, str, str]],
    factsheets: list[tuple[str, str, str]],
    selected_lens_id: str | None = None,
    selected_factsheet_ids: list[str] | None = None,
    hits_count: int,
) -> Node:
    content: list[Node] = []
    content.append(render_search_form(lenses=lenses, selected_lens_id=selected_lens_id, hits_count=hits_count))

    if corrected_query:
        selected_factsheet_ids = selected_factsheet_ids or []
        sq_params = {
            "lens_ids": [selected_lens_id] if selected_lens_id else [],
            "factsheet_ids": selected_factsheet_ids,
            "exact": True,
        }
        raw = json.dumps(sq_params, separators=(",", ":")).encode("utf-8")
        sq = base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")
        content.append(
            p[
                "Searching for ",
                strong[corrected_query],
                br(),
                "Search for ",
                a(href=f"/v1/search?q={quote(original_query)}&sq={sq}")[
                    span(class_="original-query")[original_query]
                ],
            ]
        )

    for hit in hits:
        content.append(render_search_result(hit=hit))

    return div(id="search-root")[*content]


def render_search(
    *,
    query: str,
    hits: Iterable[Any],
    corrected_query: str | None,
    original_query: str,
    lenses: list[tuple[str, str, str]],
    factsheets: list[tuple[str, str, str]],
    selected_lens_id: str | None = None,
    selected_factsheet_ids: list[str] | None = None,
    limit: int = 10,
    exact: bool = False,
) -> Node:
    hits_list = list(hits)
    return render_page(
        title_text=f"Search: {query}",
        app_data=AppData(
            search=SearchAppData(
                lens_map={at_name: lens_id for lens_id, at_name, _label in lenses if at_name},
                factsheet_map={tag: factsheet_id for factsheet_id, tag, _title in factsheets if tag},
                factsheet_title_map={tag: title for _factsheet_id, tag, title in factsheets if tag},
                factsheet_ids=selected_factsheet_ids or [],
                content=query,
                lens_ids=[selected_lens_id] if selected_lens_id else [],
                limit=limit,
                exact=exact,
            )
        ),
        content=render_search_root(
            query=query,
            hits=hits_list,
            corrected_query=corrected_query,
            original_query=original_query,
            lenses=lenses,
            factsheets=factsheets,
            selected_lens_id=selected_lens_id,
            selected_factsheet_ids=selected_factsheet_ids,
            hits_count=len(hits_list),
        ),
    )
