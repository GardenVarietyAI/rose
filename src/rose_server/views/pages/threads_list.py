import re
from datetime import datetime

from htpy import (
    Node,
    a,
    button,
    div,
    form,
    label,
    option,
    p,
    select,
    span,
    table,
    tbody,
    td,
    th,
    thead,
    tr,
)

from rose_server.schemas.threads import ThreadListItem
from rose_server.views.app_data import AppData
from rose_server.views.layout import render_page


def _format_timestamp(ts: int) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _snip(text: str | None, max_chars: int = 100) -> str:
    """Strip IDE context tags from Claude Code messages."""
    if not text:
        return ""

    text = re.sub(r"<ide_opened_file>.*?</ide_opened_file>", "", text)
    text = re.sub(r"<ide_selection>.*?</ide_selection>", "", text, flags=re.DOTALL)
    text = text.strip()

    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def render_threads_list(
    *,
    threads: list[ThreadListItem],
    total: int,
    page: int,
    limit: int,
    sort: str,
    has_assistant: str | None = None,
    import_source: str | None = None,
) -> Node:
    total_pages = (total + limit - 1) // limit

    # pagination querystring
    query_params = [f"sort={sort}", f"limit={limit}"]
    if has_assistant:
        query_params.append(f"has_assistant={has_assistant}")
    if import_source:
        query_params.append(f"import_source={import_source}")
    query_string = "&".join(query_params)

    content = div(class_="threads-list-page", x_data="threadsListPage()")[
        div(class_="threads-header")[
            div(class_="threads-stats")[
                p[
                    span[str(total)],
                    " threads total",
                ],
            ],
            div(class_="threads-filters")[
                form(method="get", action="/v1/threads", class_="filters-form")[
                    div(class_="filter-group")[
                        label(for_="sort")["Sort by:"],
                        select(name="sort", id="sort", onchange="this.form.submit()")[
                            option(value="last_activity", selected=(sort == "last_activity"))["Last Activity"],
                            option(value="created_at", selected=(sort == "created_at"))["Created"],
                        ],
                    ],
                    div(class_="filter-group")[
                        label(for_="has_assistant")["Has response:"],
                        select(name="has_assistant", id="has_assistant", onchange="this.form.submit()")[
                            option(value="", selected=(not has_assistant))["All"],
                            option(value="true", selected=(has_assistant == "true"))["Yes"],
                            option(value="false", selected=(has_assistant == "false"))["No"],
                        ],
                    ],
                    div(class_="filter-group")[
                        label(for_="import_source")["Source:"],
                        select(name="import_source", id="import_source", onchange="this.form.submit()")[
                            option(value="", selected=(not import_source))["All"],
                            option(value="claude_code_jsonl", selected=(import_source == "claude_code_jsonl"))[
                                "Imported"
                            ],
                        ],
                    ],
                ],
            ],
        ],
        div(class_="threads-table-wrapper")[
            table(class_="threads-table")[
                thead[
                    tr[
                        th["Question"],
                        th["Created"],
                        th["Last Activity"],
                        th["Status"],
                        th["Actions"],
                    ]
                ],
                tbody[
                    [
                        tr[
                            td[a(href=f"/v1/threads/{thread.thread_id}")[_snip(thread.first_message_content, 150)]],
                            td[_format_timestamp(thread.created_at)],
                            td[_format_timestamp(thread.last_activity_at)],
                            td[
                                span(
                                    class_="status-badge status-responded"
                                    if thread.has_assistant_response
                                    else "status-badge status-pending"
                                )["Responded" if thread.has_assistant_response else "No response"],
                                (
                                    span(class_="status-badge status-imported")["Imported"]
                                    if thread.import_source
                                    else None
                                ),
                            ],
                            td[
                                button(
                                    {
                                        "class": "delete-thread-btn",
                                        "@click": f"deleteThread('{thread.thread_id}')",
                                        ":disabled": f"deleting === '{thread.thread_id}'",
                                        "type": "button",
                                    }
                                )[
                                    span(x_show=f"deleting !== '{thread.thread_id}'")["Delete"],
                                    span(x_show=f"deleting === '{thread.thread_id}'", x_cloak="")["Deleting..."],
                                ]
                            ],
                        ]
                        for thread in threads
                    ]
                ],
            ]
        ],
        div(class_="pagination")[
            a(
                href=f"/v1/threads?page={page - 1}&{query_string}",
                class_="pagination-btn" + (" disabled" if page <= 1 else ""),
            )["← Previous"],
            span(class_="pagination-info")[f"Page {page} of {total_pages}"],
            a(
                href=f"/v1/threads?page={page + 1}&{query_string}",
                class_="pagination-btn" + (" disabled" if page >= total_pages else ""),
            )["Next →"],
        ],
    ]

    return render_page(title_text="Threads - ROSE", app_data=AppData(), content=content)
