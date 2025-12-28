from typing import Annotated

from fastapi import Form
from pydantic import BaseModel, StringConstraints, field_validator


class CreateLensRequest(BaseModel):
    at_name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^[A-Za-z0-9]+$"),
    ]
    label: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    system_prompt: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    @field_validator("at_name")
    @classmethod
    def validate_at_name(cls, value: str) -> str:
        return value.lower()

    @classmethod
    def as_form(
        cls,
        at_name: str = Form(...),
        label: str = Form(...),
        system_prompt: str = Form(...),
    ) -> "CreateLensRequest":
        return cls(at_name=at_name, label=label, system_prompt=system_prompt)
