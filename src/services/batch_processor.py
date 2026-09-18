import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.entities import ImageModel, EmbeddingModel
from src.services.vision_service import VisionService, InvalidVisionOutputError
from src.services.embedding_service import EmbeddingService
from src.services.cost_tracker import CostTracker
from src.config import settings

logger = logging.getLogger(__name__)

class BatchJobState:
    def __init__(self):
        self.job_id: str = "batch_default"
        self.status: str = "idle"  # idle, running, completed, failed
        self.total_images: int = 0
        self.processed_count: int = 0
        self.flagged_count: int = 0
        self.failed_count: int = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.errors: List[Dict[str, str]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "total_images": self.total_images,
            "processed_count": self.processed_count,
            "flagged_count": self.flagged_count,
            "failed_count": self.failed_count,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "errors": self.errors
        }

current_batch_state = BatchJobState()

class BatchProcessor:
    @classmethod
    def get_current_state(cls) -> Dict[str, Any]:
        return current_batch_state.to_dict()

    @classmethod
    async def process_corpus(
        cls,
        session: AsyncSession,
        corpus_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes image corpus in the background with:
        - Exponential backoff retries
        - Low-confidence flagging
        - Per-call cost tracking
        - Idempotent execution (skips already processed images)
        """
        directory = Path(corpus_dir or settings.IMAGE_CORPUS_DIR)
        if not directory.exists():
            raise FileNotFoundError(f"Corpus directory not found: {directory}")

        image_files = sorted([
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        ])

        current_batch_state.status = "running"
        current_batch_state.total_images = len(image_files)
        current_batch_state.processed_count = 0
        current_batch_state.flagged_count = 0
        current_batch_state.failed_count = 0
        current_batch_state.start_time = datetime.now(timezone.utc)
        current_batch_state.errors.clear()

        logger.info(f"Starting batch ingestion of {len(image_files)} images from {directory}")

        for img_path in image_files:
            filename = img_path.name

            # Check idempotency: check if image already exists in database
            existing_query = select(ImageModel).where(ImageModel.filename == filename)
            existing_res = await session.execute(existing_query)
            existing_img = existing_res.scalar_one_or_none()

            if existing_img is not None:
                logger.debug(f"Image {filename} already exists in DB. Skipping to preserve idempotency.")
                if existing_img.status == "flagged_low_confidence":
                    current_batch_state.flagged_count += 1
                current_batch_state.processed_count += 1
                continue

            # Process with retries
            attempt = 0
            success = False
            last_error = ""

            while attempt < settings.MAX_BATCH_RETRIES and not success:
                attempt += 1
                try:
                    metadata, status, flag_reason = await VisionService.analyze_image(str(img_path))

                    # Track vision cost
                    await CostTracker.record_cost(
                        session=session,
                        operation="vision_classification",
                        model=settings.VISION_MODEL,
                        input_tokens=258,
                        output_tokens=64,
                        item_identifier=filename
                    )

                    # Persist Image Record
                    new_image = ImageModel(
                        filename=filename,
                        filepath=str(img_path),
                        category=metadata.category,
                        subject=metadata.subject,
                        attributes=metadata.attributes,
                        caption=metadata.caption,
                        confidence=metadata.confidence,
                        status=status,
                        flag_reason=flag_reason
                    )
                    session.add(new_image)
                    await session.flush()

                    # Generate and store embedding for caption
                    embedding_vector = await EmbeddingService.get_embedding(metadata.caption)
                    
                    # Track embedding cost
                    await CostTracker.record_cost(
                        session=session,
                        operation="text_embedding",
                        model=settings.EMBEDDING_MODEL,
                        input_tokens=len(metadata.caption.split()),
                        output_tokens=0,
                        item_identifier=f"emb_{filename}"
                    )

                    new_embedding = EmbeddingModel(
                        entity_type="image",
                        entity_id=new_image.id,
                        vector=embedding_vector,
                        dimension=len(embedding_vector),
                        model=settings.EMBEDDING_MODEL
                    )
                    session.add(new_embedding)
                    await session.flush()

                    current_batch_state.processed_count += 1
                    if status == "flagged_low_confidence":
                        current_batch_state.flagged_count += 1

                    success = True
                    logger.info(f"[{current_batch_state.processed_count}/{current_batch_state.total_images}] Ingested {filename} -> status: {status}")

                except Exception as ex:
                    last_error = str(ex)
                    logger.warning(f"Attempt {attempt} failed for {filename}: {ex}")
                    if attempt < settings.MAX_BATCH_RETRIES:
                        await asyncio.sleep(settings.BATCH_RETRY_DELAY_SEC * (2 ** (attempt - 1)))

            if not success:
                current_batch_state.failed_count += 1
                current_batch_state.errors.append({"filename": filename, "error": last_error})
                logger.error(f"Image {filename} failed after {settings.MAX_BATCH_RETRIES} attempts: {last_error}")

        await session.commit()
        current_batch_state.status = "completed"
        current_batch_state.end_time = datetime.now(timezone.utc)
        logger.info(f"Batch ingestion completed: {current_batch_state.to_dict()}")
        return current_batch_state.to_dict()
