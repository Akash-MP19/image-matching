from typing import List, Optional
from pydantic import BaseModel, Field

class SuggestedCandidate(BaseModel):
    suggestion_id: Optional[str] = None
    image_id: str
    filename: str
    category: str
    subject: str
    caption: str
    attributes: List[str] = Field(default_factory=list)
    similarity_score: float
    guard_status: str  # 'APPROVED' or 'REJECTED'
    rejection_reason: Optional[str] = None
    explanation: Optional[str] = None

class PostMatchResponse(BaseModel):
    post_id: str
    post_title: str
    status: str  # 'MATCH_FOUND' or 'NO_CONFIDENT_MATCH'
    suggested_image: Optional[SuggestedCandidate] = None
    candidates: List[SuggestedCandidate] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)
    decision_summary: str
