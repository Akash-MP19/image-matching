from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict

class ReviewCreate(BaseModel):
    decision: str = Field(..., description="'APPROVED' or 'REJECTED'")
    notes: Optional[str] = Field(None, description="Optional feedback or rationale")

    @field_validator("decision")
    @classmethod
    def validate_decision(cls, v: str) -> str:
        upper = v.strip().upper()
        if upper not in ("APPROVED", "REJECTED"):
            raise ValueError("Decision must be either 'APPROVED' or 'REJECTED'")
        return upper

class ReviewResponse(BaseModel):
    id: str
    suggestion_id: str
    post_id: str
    image_id: str
    decision: str
    notes: Optional[str] = None
    reviewed_at: datetime

    model_config = ConfigDict(from_attributes=True)
