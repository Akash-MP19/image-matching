import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.entities import CostLogModel
from src.config import settings

logger = logging.getLogger(__name__)

class BudgetExceededError(Exception):
    """Raised when total AI expenditure exceeds budget limit."""
    pass

class CostTracker:
    @staticmethod
    async def get_total_spend(session: AsyncSession) -> float:
        query = select(func.coalesce(func.sum(CostLogModel.estimated_cost_usd), 0.0))
        result = await session.execute(query)
        return float(result.scalar() or 0.0)

    @staticmethod
    async def record_cost(
        session: AsyncSession,
        operation: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
        item_identifier: Optional[str] = None
    ) -> CostLogModel:
        # Calculate cost if not explicitly passed
        if estimated_cost_usd is None:
            if "vision" in operation.lower():
                estimated_cost_usd = settings.COST_PER_VISION_CALL_USD
            else:
                total_tokens = input_tokens + output_tokens
                estimated_cost_usd = (total_tokens / 1000.0) * settings.COST_PER_1K_EMBEDDING_TOKENS_USD

        # Check budget limit before recording
        current_spend = await CostTracker.get_total_spend(session)
        if current_spend + estimated_cost_usd > settings.BUDGET_LIMIT_USD:
            logger.warning(
                f"Budget limit exceeded! Current spend: ${current_spend:.5f}, "
                f"Attempted cost: ${estimated_cost_usd:.5f}, Limit: ${settings.BUDGET_LIMIT_USD:.2f}"
            )
            # Budget guard stops runaway costs
            raise BudgetExceededError(
                f"Budget limit of ${settings.BUDGET_LIMIT_USD:.2f} exceeded. Current spend: ${current_spend:.5f}"
            )

        cost_entry = CostLogModel(
            operation=operation,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=round(estimated_cost_usd, 6),
            item_identifier=item_identifier,
            timestamp=datetime.now(timezone.utc)
        )
        session.add(cost_entry)
        await session.flush()
        return cost_entry

    @staticmethod
    async def get_summary(session: AsyncSession) -> Dict[str, Any]:
        total_query = select(
            func.count(CostLogModel.id),
            func.coalesce(func.sum(CostLogModel.estimated_cost_usd), 0.0)
        )
        total_res = await session.execute(total_query)
        total_calls, total_cost = total_res.first() or (0, 0.0)
        total_cost = float(total_cost)

        by_op_query = select(
            CostLogModel.operation,
            func.coalesce(func.sum(CostLogModel.estimated_cost_usd), 0.0)
        ).group_by(CostLogModel.operation)
        by_op_res = await session.execute(by_op_query)
        by_operation = {op: round(float(cost), 6) for op, cost in by_op_res.all()}

        budget_remaining = max(0.0, settings.BUDGET_LIMIT_USD - total_cost)

        return {
            "total_cost_usd": round(total_cost, 6),
            "total_calls": int(total_calls),
            "by_operation": by_operation,
            "budget_limit_usd": settings.BUDGET_LIMIT_USD,
            "budget_remaining_usd": round(budget_remaining, 6),
            "budget_exceeded": total_cost >= settings.BUDGET_LIMIT_USD
        }
