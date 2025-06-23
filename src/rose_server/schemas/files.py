from typing import List

from openai.types import FileObject
from pydantic import BaseModel, Field


class FileListResponse(BaseModel):
    object: str = Field(default="list", description="Object type")
    data: List[FileObject] = Field(default_factory=list, description="List of files")
    has_more: bool = Field(default=False, description="Whether there are more files")
