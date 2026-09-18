import pytest
from pathlib import Path
from sqlalchemy import select
from src.database import AsyncSessionLocal
from src.models.entities import ImageModel
from src.services.batch_processor import BatchProcessor
from src.services.vision_service import VisionService

@pytest.mark.asyncio
async def test_low_confidence_flagging_logic():
    metadata, status, flag_reason = await VisionService.analyze_image("blurry_mystery_01.jpg")
    assert status == "flagged_low_confidence"
    assert metadata.confidence < 0.70
    assert flag_reason is not None

@pytest.mark.asyncio
async def test_batch_processor_idempotency():
    async with AsyncSessionLocal() as session:
        # Run 1
        res1 = await BatchProcessor.process_corpus(session)
        count1 = len((await session.execute(select(ImageModel))).scalars().all())

        # Run 2 (re-run)
        res2 = await BatchProcessor.process_corpus(session)
        count2 = len((await session.execute(select(ImageModel))).scalars().all())

        # Assert no duplicates were created
        assert count1 == count2, f"Idempotency violated: run 1 had {count1}, run 2 had {count2}"
        assert count1 >= 40
