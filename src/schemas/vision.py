from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict

class VisionMetadata(BaseModel):
    subject: str = Field(..., description="Primary subject of the image (e.g., 'red fox')")
    category: str = Field(..., description="High-level category (e.g., 'animal', 'landscape')")
    attributes: List[str] = Field(default_factory=list, description="Descriptive visual tags/attributes")
    caption: str = Field(..., description="Detailed factual caption describing the image content")
    confidence: float = Field(..., description="Model confidence score between 0.0 and 1.0")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, received {v}")
        return round(v, 4)

    @field_validator("subject", "category", "caption")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

class ImageResponse(BaseModel):
    id: str
    filename: str
    filepath: str
    category: str
    subject: str
    attributes: List[str]
    caption: str
    confidence: float
    status: str
    flag_reason: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ImageIngestResponse(BaseModel):
    id: str
    filename: str
    status: str
    category: str
    subject: str
    confidence: float
    flag_reason: Optional[str] = None
    message: str
