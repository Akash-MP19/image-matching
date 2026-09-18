from typing import List
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.models.entities import CostLogModel
from src.schemas.cost import CostSummaryResponse, CostLogResponse
from src.services.cost_tracker import CostTracker

router = APIRouter(prefix="/costs", tags=["Cost Tracking & Budget Guard"])

@router.get("", response_model=CostSummaryResponse)
async def get_cost_summary(session: AsyncSession = Depends(get_db)):
    """Retrieve aggregate AI cost summary, call counts, and budget tracking."""
    summary = await CostTracker.get_summary(session)
    return summary

@router.get("/ledger", response_model=List[CostLogResponse])
async def get_cost_ledger(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_db)
):
    """
    Retrieve itemized cost ledger.
    Every vision classification and text embedding call is attributed with cost and tokens.
    """
    query = select(CostLogModel).order_by(CostLogModel.timestamp.desc()).offset(offset).limit(limit)
    res = await session.execute(query)
    return res.scalars().all()
