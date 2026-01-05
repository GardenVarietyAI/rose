from typing import Annotated

from fastapi import Form
from pydantic import BaseModel, StringConstraints, field_validator


class CreateFactsheetRequest(BaseModel):
    tag: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^[A-Za-z0-9]+$"),
    ]
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    body: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    @field_validator("tag")
    @classmethod
    def normalize_tag(cls, value: str) -> str:
        return value.lower()

    @classmethod
    def as_form(
        cls,
        tag: str = Form(...),
        title: str = Form(...),
        body: str = Form(...),
    ) -> "CreateFactsheetRequest":
        return cls(tag=tag, title=title, body=body)
