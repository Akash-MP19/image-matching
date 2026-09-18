import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status, Body
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.models.entities import PostModel, ImageModel, EmbeddingModel, SuggestionModel
from src.schemas.post import PostCreate, PostResponse
from src.schemas.suggestion import PostMatchResponse, SuggestedCandidate
from src.services.embedding_service import EmbeddingService
from src.services.cost_tracker import CostTracker
from src.guard.mismatch_guard import MismatchGuard
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/posts", tags=["Posts & Matching"])

@router.post("", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def create_post(payload: PostCreate, session: AsyncSession = Depends(get_db)):
    """Create a new article / post and generate its semantic embedding."""
    new_post = PostModel(
        title=payload.title,
        content=payload.content,
        expected_category=payload.expected_category,
        ground_truth_image_id=payload.ground_truth_image_id
    )
    session.add(new_post)
    await session.flush()

    # Generate embedding for post title and content combined
    combined_text = f"{new_post.title}. {new_post.content}"
    vector = await EmbeddingService.get_embedding(combined_text)

    # Track embedding cost
    await CostTracker.record_cost(
        session=session,
        operation="text_embedding",
        model=settings.EMBEDDING_MODEL,
        input_tokens=len(combined_text.split()),
        output_tokens=0,
        item_identifier=f"post_{new_post.id}"
    )

    new_embedding = EmbeddingModel(
        entity_type="post",
        entity_id=new_post.id,
        vector=vector,
        dimension=len(vector),
        model=settings.EMBEDDING_MODEL
    )
    session.add(new_embedding)
    await session.commit()
    await session.refresh(new_post)
    return new_post

@router.get("", response_model=List[PostResponse])
async def list_posts(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_db)
):
    """List all registered posts."""
    query = select(PostModel).order_by(PostModel.created_at.desc()).offset(offset).limit(limit)
    result = await session.execute(query)
    return result.scalars().all()

@router.get("/{post_id}", response_model=PostResponse)
async def get_post(post_id: str, session: AsyncSession = Depends(get_db)):
    """Retrieve post details by ID."""
    query = select(PostModel).where(PostModel.id == post_id)
    result = await session.execute(query)
    post = result.scalar_one_or_none()
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Post with ID '{post_id}' not found"
        )
    return post

@router.get("/{post_id}/images", response_model=PostMatchResponse)
async def get_matching_images_for_post(
    post_id: str,
    top_k: int = Query(5, ge=1, le=20, description="Number of candidate images to evaluate"),
    session: AsyncSession = Depends(get_db)
):
    """
    Query images for a post.
    Computes cosine similarity ranking and passes all candidates through the Mismatch Guard.
    Guarantees that incorrect pairings (e.g. wolf for fox) are rejected with clear explanations.
    """
    # 1. Fetch Post
    post_query = select(PostModel).where(PostModel.id == post_id)
    post_res = await session.execute(post_query)
    post = post_res.scalar_one_or_none()
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Post with ID '{post_id}' not found"
        )

    # 2. Fetch Post Embedding
    post_emb_query = select(EmbeddingModel).where(
        EmbeddingModel.entity_type == "post",
        EmbeddingModel.entity_id == post_id
    )
    post_emb_res = await session.execute(post_emb_query)
    post_emb = post_emb_res.scalar_one_or_none()

    if not post_emb:
        # Fallback: compute on the fly
        combined_text = f"{post.title}. {post.content}"
        vector = await EmbeddingService.get_embedding(combined_text)
        post_emb = EmbeddingModel(
            entity_type="post",
            entity_id=post.id,
            vector=vector,
            dimension=len(vector),
            model=settings.EMBEDDING_MODEL
        )
        session.add(post_emb)
        await session.flush()

    # 3. Fetch Image Embeddings & Images
    img_emb_query = select(EmbeddingModel, ImageModel).join(
        ImageModel, EmbeddingModel.entity_id == ImageModel.id
    ).where(EmbeddingModel.entity_type == "image")
    img_emb_res = await session.execute(img_emb_query)
    rows = img_emb_res.all()

    if not rows:
        return PostMatchResponse(
            post_id=post.id,
            post_title=post.title,
            status="NO_CONFIDENT_MATCH",
            suggested_image=None,
            candidates=[],
            rejection_reasons=["No images currently available in the database."],
            decision_summary="Image library is empty."
        )

    # 4. Rank Candidates by Cosine Similarity
    scored_candidates = []
    for emb, img in rows:
        sim = EmbeddingService.cosine_similarity(post_emb.vector, emb.vector)
        scored_candidates.append((img, sim))

    # Sort descending by similarity score
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scored_candidates[:top_k]

    # 5. Evaluate through Mismatch Guard
    match_response = MismatchGuard.filter_and_rank(post, top_candidates)

    # 6. Persist suggestions for audit & review
    for rank, candidate in enumerate(match_response.candidates, start=1):
        # Check if suggestion already recorded for this post & image
        existing_sug_query = select(SuggestionModel).where(
            SuggestionModel.post_id == post.id,
            SuggestionModel.image_id == candidate.image_id
        )
        existing_sug = (await session.execute(existing_sug_query)).scalar_one_or_none()

        if existing_sug:
            existing_sug.rank = rank
            existing_sug.similarity_score = candidate.similarity_score
            existing_sug.guard_status = candidate.guard_status
            existing_sug.rejection_reason = candidate.rejection_reason
            candidate.suggestion_id = existing_sug.id
        else:
            new_suggestion = SuggestionModel(
                post_id=post.id,
                image_id=candidate.image_id,
                rank=rank,
                similarity_score=candidate.similarity_score,
                guard_status=candidate.guard_status,
                rejection_reason=candidate.rejection_reason
            )
            session.add(new_suggestion)
            await session.flush()
            candidate.suggestion_id = new_suggestion.id

    if match_response.suggested_image:
        for c in match_response.candidates:
            if c.image_id == match_response.suggested_image.image_id:
                match_response.suggested_image.suggestion_id = c.suggestion_id
                break

    await session.commit()
    return match_response

@router.post("/{post_id}/force-candidate", response_model=SuggestedCandidate)
async def force_candidate_evaluation(
    post_id: str,
    image_identifier: str = Body(..., embed=True, description="Image ID or Filename to force as candidate"),
    session: AsyncSession = Depends(get_db)
):
    """
    Forces an explicit image as a candidate for a post to test the Mismatch Guard.
    Provably confirms that invalid pairings (such as wolf on fox post) are rejected.
    """
    post_query = select(PostModel).where(PostModel.id == post_id)
    post = (await session.execute(post_query)).scalar_one_or_none()
    if not post:
        raise HTTPException(status_code=404, detail=f"Post '{post_id}' not found")

    # Find image by ID or filename
    img_query = select(ImageModel).where(
        (ImageModel.id == image_identifier) | (ImageModel.filename == image_identifier)
    )
    image = (await session.execute(img_query)).scalar_one_or_none()
    if not image:
        raise HTTPException(status_code=404, detail=f"Image '{image_identifier}' not found")

    # Get embeddings and similarity
    post_emb_q = select(EmbeddingModel).where(EmbeddingModel.entity_type == "post", EmbeddingModel.entity_id == post.id)
    post_emb = (await session.execute(post_emb_q)).scalar_one_or_none()

    img_emb_q = select(EmbeddingModel).where(EmbeddingModel.entity_type == "image", EmbeddingModel.entity_id == image.id)
    img_emb = (await session.execute(img_emb_q)).scalar_one_or_none()

    if not post_emb:
        post_vec = await EmbeddingService.get_embedding(f"{post.title}. {post.content}")
    else:
        post_vec = post_emb.vector

    if not img_emb:
        img_vec = await EmbeddingService.get_embedding(image.caption)
    else:
        img_vec = img_emb.vector

    sim = EmbeddingService.cosine_similarity(post_vec, img_vec)

    is_approved, explanation = MismatchGuard.evaluate_candidate(
        post_title=post.title,
        post_content=post.content,
        candidate_image=image,
        similarity_score=sim,
        post_expected_category=post.expected_category
    )

    # Persist suggestion
    suggestion = SuggestionModel(
        post_id=post.id,
        image_id=image.id,
        rank=1,
        similarity_score=sim,
        guard_status="APPROVED" if is_approved else "REJECTED",
        rejection_reason=explanation if not is_approved else None
    )
    session.add(suggestion)
    await session.commit()
    await session.refresh(suggestion)

    return SuggestedCandidate(
        suggestion_id=suggestion.id,
        image_id=image.id,
        filename=image.filename,
        category=image.category,
        subject=image.subject,
        caption=image.caption,
        attributes=image.attributes or [],
        similarity_score=sim,
        guard_status=suggestion.guard_status,
        rejection_reason=suggestion.rejection_reason,
        explanation=explanation if is_approved else None
    )
