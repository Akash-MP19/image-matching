from typing import Dict, Optional
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class CostLogResponse(BaseModel):
    id: str
    operation: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    item_identifier: Optional[str] = None
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)

class CostSummaryResponse(BaseModel):
    total_cost_usd: float
    total_calls: int
    by_operation: Dict[str, float]
    budget_limit_usd: float
    budget_remaining_usd: float
    budget_exceeded: bool
