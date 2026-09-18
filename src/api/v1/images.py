from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.models.entities import ImageModel
from src.schemas.vision import ImageResponse

router = APIRouter(prefix="/images", tags=["Images"])

@router.get("", response_model=List[ImageResponse])
async def list_images(
    category: Optional[str] = Query(None, description="Filter by category (e.g. animal)"),
    status: Optional[str] = Query(None, description="Filter by status (processed, flagged_low_confidence)"),
    subject: Optional[str] = Query(None, description="Filter by subject (e.g. red fox)"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_db)
):
    """List ingested images with their structured metadata and quality status."""
    query = select(ImageModel)
    if category:
        query = query.where(ImageModel.category == category)
    if status:
        query = query.where(ImageModel.status == status)
    if subject:
        query = query.where(ImageModel.subject == subject)

    query = query.order_by(ImageModel.filename.asc()).offset(offset).limit(limit)
    result = await session.execute(query)
    images = result.scalars().all()
    return images

@router.get("/{image_id}", response_model=ImageResponse)
async def get_image(image_id: str, session: AsyncSession = Depends(get_db)):
    """Retrieve metadata for a single image by ID."""
    query = select(ImageModel).where(ImageModel.id == image_id)
    result = await session.execute(query)
    image = result.scalar_one_or_none()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with ID '{image_id}' not found"
        )
    return image
