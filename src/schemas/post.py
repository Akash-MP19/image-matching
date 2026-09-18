from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict

class PostCreate(BaseModel):
    title: str = Field(..., description="Title of the article / blog post")
    content: str = Field(..., description="Full body text of the article")
    expected_category: Optional[str] = Field(None, description="Expected semantic category or entity")
    ground_truth_image_id: Optional[str] = Field(None, description="Known correct image ID for evaluation")

    @field_validator("title", "content")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

class PostResponse(BaseModel):
    id: str
    title: str
    content: str
    expected_category: Optional[str] = None
    ground_truth_image_id: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
