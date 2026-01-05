from typing import Any, Literal, TypeVar, cast

from sqlalchemy import (
    case,
    func,
    select as sa_select,
)
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import CTE, Select
from sqlmodel import col

from rose_server.models.messages import Message

_TSelect = TypeVar("_TSelect", bound=Select[Any])


def _col(expr: Any, /) -> ColumnElement[Any]:
    return cast(ColumnElement[Any], col(expr))


def _threads_ctes() -> tuple[CTE, CTE]:
    first_messages = (
        sa_select(
            _col(Message.thread_id).label("thread_id"),
            _col(Message.content).label("first_message_content"),
            _col(Message.role).label("first_message_role"),
            _col(Message.created_at).label("created_at"),
            _col(Message.import_source).label("import_source"),
            func.row_number()
            .over(
                partition_by=_col(Message.thread_id),
                order_by=(_col(Message.created_at).asc(), _col(Message.id).asc()),
            )
            .label("rn"),
        )
        .where(
            _col(Message.deleted_at).is_(None),
            _col(Message.thread_id).is_not(None),
        )
        .cte("thread_first_messages")
    )

    stats = (
        sa_select(
            _col(Message.thread_id).label("thread_id"),
            func.max(_col(Message.created_at)).label("last_activity_at"),
            func.max(case((_col(Message.role) == "assistant", 1), else_=0)).label("has_assistant_response"),
        )
        .where(
            _col(Message.deleted_at).is_(None),
            _col(Message.thread_id).is_not(None),
        )
        .group_by(_col(Message.thread_id))
        .cte("thread_stats")
    )

    return first_messages, stats


def _apply_threads_filters(
    stmt: _TSelect,
    *,
    first_messages: CTE,
    stats: CTE,
    date_from: int | None,
    date_to: int | None,
    has_assistant: bool | None,
    import_source: str | None,
) -> _TSelect:
    if date_from is not None:
        stmt = stmt.where(first_messages.c.created_at >= date_from)

    if date_to is not None:
        stmt = stmt.where(first_messages.c.created_at <= date_to)

    if has_assistant is not None:
        stmt = stmt.where(stats.c.has_assistant_response == (1 if has_assistant else 0))

    if import_source is not None:
        stmt = stmt.where(first_messages.c.import_source == import_source)

    return stmt


def build_threads_list_statements(
    *,
    sort: Literal["last_activity", "created_at"],
    page: int,
    limit: int,
    date_from: int | None,
    date_to: int | None,
    has_assistant: bool | None,
    import_source: str | None,
) -> tuple[Select[Any], Select[Any]]:
    first_messages, stats = _threads_ctes()

    base_stmt = (
        sa_select(
            first_messages.c.thread_id,
            first_messages.c.first_message_content,
            first_messages.c.first_message_role,
            first_messages.c.created_at,
            stats.c.last_activity_at,
            stats.c.has_assistant_response,
            first_messages.c.import_source,
        )
        .select_from(first_messages.join(stats, first_messages.c.thread_id == stats.c.thread_id))
        .where(first_messages.c.rn == 1)
    )

    base_stmt = _apply_threads_filters(
        base_stmt,
        first_messages=first_messages,
        stats=stats,
        date_from=date_from,
        date_to=date_to,
        has_assistant=has_assistant,
        import_source=import_source,
    )

    sort_col = stats.c.last_activity_at if sort == "last_activity" else first_messages.c.created_at
    offset = (page - 1) * limit
    list_stmt = base_stmt.order_by(sort_col.desc()).limit(limit).offset(offset)

    count_stmt = (
        sa_select(func.count())
        .select_from(first_messages.join(stats, first_messages.c.thread_id == stats.c.thread_id))
        .where(first_messages.c.rn == 1)
    )
    count_stmt = _apply_threads_filters(
        count_stmt,
        first_messages=first_messages,
        stats=stats,
        date_from=date_from,
        date_to=date_to,
        has_assistant=has_assistant,
        import_source=import_source,
    )

    return list_stmt, count_stmt
