from htpy import BaseElement, Node, div


def message_card(
    *,
    uuid: str | None,
    dom_id: str | None,
    accepted: bool = False,
    root_attrs: dict[str, str] | None = None,
    header_children: list[Node],
    body_children: list[Node],
    body_attrs: dict[str, str] | None = None,
    extra_class: str | None = None,
    x_data: str | None = None,
) -> BaseElement:
    attrs: dict[str, str] = dict(root_attrs or {})
    if dom_id:
        attrs["id"] = dom_id

    classes = ["message"]
    if accepted:
        classes.append("accepted")
    if extra_class:
        classes.append(extra_class)

    kwargs: dict[str, str] = {"class_": " ".join(classes)}
    if uuid:
        kwargs["data_uuid"] = uuid
    if x_data:
        kwargs["x_data"] = x_data

    attributes = dict(body_attrs or {})
    body_class = str(attributes.get("class") or "").strip()
    attributes["class"] = "message-body" if not body_class else f"message-body {body_class}"

    return div(attrs, **kwargs)[
        div(class_="message-header")[*header_children],
        div(attributes)[*body_children],
    ]
