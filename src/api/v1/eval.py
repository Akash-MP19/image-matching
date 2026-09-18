import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.models.entities import PostModel, ImageModel, EmbeddingModel
from src.services.embedding_service import EmbeddingService
from src.guard.mismatch_guard import MismatchGuard
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["Evaluation & Quality"])

latest_eval_result: Optional[Dict[str, Any]] = None

@router.post("/run")
async def run_evaluation(session: AsyncSession = Depends(get_db)):
    """
    Executes the evaluation benchmark on the labeled dataset.
    Measures Top-1 Precision: the fraction of posts where the rank-1 approved
    recommendation exactly matches the ground-truth labeled image.
    """
    global latest_eval_result
    eval_file = Path(settings.EVAL_POSTS_FILE)
    if not eval_file.exists():
        raise HTTPException(status_code=404, detail=f"Eval posts file not found at {eval_file}")

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    labeled_posts: List[Dict[str, Any]] = eval_data.get("posts", [])
    if not labeled_posts:
        raise HTTPException(status_code=400, detail="Eval dataset is empty")

    # Fetch all image embeddings
    img_emb_query = select(EmbeddingModel, ImageModel).join(
        ImageModel, EmbeddingModel.entity_id == ImageModel.id
    ).where(EmbeddingModel.entity_type == "image")
    img_emb_res = await session.execute(img_emb_query)
    image_rows = img_emb_res.all()

    if not image_rows:
        raise HTTPException(status_code=400, detail="No images found in database. Ingest corpus first.")

    total_posts = len(labeled_posts)
    top1_correct = 0
    safe_rejections = 0
    eval_breakdown = []

    for item in labeled_posts:
        title = item["title"]
        content = item["content"]
        expected_cat = item.get("expected_category")
        gt_filename = item.get("ground_truth_filename")
        is_negative_example = item.get("is_negative_example", False)

        # Generate embedding for post
        combined_text = f"{title}. {content}"
        post_vec = await EmbeddingService.get_embedding(combined_text)

        # Rank candidates
        scored = []
        for emb, img in image_rows:
            sim = EmbeddingService.cosine_similarity(post_vec, emb.vector)
            scored.append((img, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored[:5]

        # Evaluate through Mismatch Guard
        dummy_post = PostModel(
            id=f"eval_{item['id']}",
            title=title,
            content=content,
            expected_category=expected_cat
        )
        match_res = MismatchGuard.filter_and_rank(dummy_post, top_candidates)

        is_correct = False
        outcome = "FAIL"

        if is_negative_example:
            # Negative test: system should refuse (NO_CONFIDENT_MATCH)
            if match_res.status == "NO_CONFIDENT_MATCH":
                is_correct = True
                safe_rejections += 1
                outcome = "SAFE_REJECTION_PASS"
            else:
                outcome = "FALSE_POSITIVE_FAIL"
        else:
            # Positive test: suggested image filename must match ground truth filename
            if match_res.suggested_image and match_res.suggested_image.filename == gt_filename:
                is_correct = True
                top1_correct += 1
                outcome = "TOP1_HIT_PASS"
            elif match_res.suggested_image:
                outcome = f"WRONG_MATCH ({match_res.suggested_image.filename} != {gt_filename})"
            else:
                outcome = "UNEXPECTED_REJECTION"

        eval_breakdown.append({
            "post_id": item["id"],
            "title": title,
            "expected_category": expected_cat,
            "ground_truth_filename": gt_filename,
            "is_negative_example": is_negative_example,
            "status": match_res.status,
            "selected_filename": match_res.suggested_image.filename if match_res.suggested_image else None,
            "similarity": match_res.suggested_image.similarity_score if match_res.suggested_image else None,
            "outcome": outcome,
            "is_correct": is_correct
        })

    # Overall precision: (top1_correct + safe_rejections) / total_posts
    successful_cases = top1_correct + safe_rejections
    precision = round(successful_cases / total_posts, 4)

    latest_eval_result = {
        "total_posts": total_posts,
        "positive_tests": total_posts - sum(1 for p in labeled_posts if p.get("is_negative_example")),
        "negative_tests": sum(1 for p in labeled_posts if p.get("is_negative_example")),
        "top1_matches": top1_correct,
        "safe_rejections": safe_rejections,
        "overall_accuracy_precision": precision,
        "precision_percentage": f"{precision * 100:.1f}%",
        "breakdown": eval_breakdown
    }

    return latest_eval_result

@router.get("/results")
async def get_latest_results():
    """Retrieve the cached result of the latest evaluation run."""
    if latest_eval_result is None:
        raise HTTPException(status_code=404, detail="No evaluation has been run yet. Call POST /api/v1/eval/run")
    return latest_eval_result
