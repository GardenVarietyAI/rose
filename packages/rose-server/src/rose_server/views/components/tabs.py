from dataclasses import dataclass

from htpy import Node, a, div, span


@dataclass(frozen=True)
class Tab:
    label: str
    href: str
    active: bool
    badge: str | None = None


def render_tabs(*, tabs: list[Tab]) -> Node:
    return div(class_="tabs")[
        *[
            a(
                href=tab.href,
                class_=("tab tab-active" if tab.active else "tab"),
            )[
                span(class_="tab-label")[tab.label],
                span(class_="tab-badge")[tab.badge] if tab.badge else "",
            ]
            for tab in tabs
        ]
    ]
