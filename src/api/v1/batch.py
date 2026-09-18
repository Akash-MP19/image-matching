import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, BackgroundTasks, Query, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db, AsyncSessionLocal
from src.services.batch_processor import BatchProcessor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch", tags=["Batch Processing"])

async def run_batch_task():
    async with AsyncSessionLocal() as session:
        try:
            await BatchProcessor.process_corpus(session)
        except Exception as e:
            logger.error(f"Background batch processing error: {e}")

@router.post("/ingest", response_model=Dict[str, Any])
async def trigger_batch_ingest(
    background_tasks: BackgroundTasks,
    wait: bool = Query(False, description="Whether to wait synchronously for batch completion"),
    session: AsyncSession = Depends(get_db)
):
    """
    Trigger batch processing of the image corpus with structured vision output,
    validation, retries, and per-call cost tracking.
    """
    if wait:
        result = await BatchProcessor.process_corpus(session)
        return result
    else:
        background_tasks.add_task(run_batch_task)
        return {
            "message": "Batch ingestion job dispatched to background",
            "state": BatchProcessor.get_current_state()
        }

@router.get("/status", response_model=Dict[str, Any])
async def get_batch_status():
    """Retrieve current background batch progress and statistics."""
    return BatchProcessor.get_current_state()
