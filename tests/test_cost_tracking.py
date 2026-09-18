import pytest
from src.database import AsyncSessionLocal
from src.services.cost_tracker import CostTracker, BudgetExceededError
from src.config import settings

@pytest.mark.asyncio
async def test_record_and_summarize_cost():
    async with AsyncSessionLocal() as session:
        entry = await CostTracker.record_cost(
            session=session,
            operation="vision_classification",
            model="gemini-2.0-flash",
            input_tokens=258,
            output_tokens=64,
            item_identifier="test_img_01.jpg"
        )
        assert entry.id is not None
        assert entry.estimated_cost_usd > 0

        summary = await CostTracker.get_summary(session)
        assert summary["total_calls"] >= 1
        assert summary["total_cost_usd"] > 0
        assert not summary["budget_exceeded"]

@pytest.mark.asyncio
async def test_budget_limit_guard():
    async with AsyncSessionLocal() as session:
        # Attempting an operation exceeding $5.00 limit should trigger BudgetExceededError
        with pytest.raises(BudgetExceededError):
            await CostTracker.record_cost(
                session=session,
                operation="vision_classification",
                model="expensive-model",
                estimated_cost_usd=10.00
            )
