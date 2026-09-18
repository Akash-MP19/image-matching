import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Text,
    DateTime,
    ForeignKey,
    Index,
    JSON
)
from sqlalchemy.orm import relationship

from src.database import Base

def generate_uuid() -> str:
    return str(uuid.uuid4())

class ImageModel(Base):
    __tablename__ = "images"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    filename = Column(String(255), unique=True, nullable=False, index=True)
    filepath = Column(String(512), nullable=False)
    category = Column(String(100), nullable=False, index=True)
    subject = Column(String(100), nullable=False, index=True)
    attributes = Column(JSON, nullable=False, default=list)
    caption = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    status = Column(String(50), nullable=False, default="processed", index=True)
    flag_reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    suggestions = relationship("SuggestionModel", back_populates="image", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_images_category_subject", "category", "subject"),
    )

class EmbeddingModel(Base):
    __tablename__ = "embeddings"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    entity_type = Column(String(20), nullable=False, index=True)  # 'image' or 'post'
    entity_id = Column(String(36), nullable=False, index=True)
    vector = Column(JSON, nullable=False)  # Stored as JSON array of floats
    dimension = Column(Integer, nullable=False, default=384)
    model = Column(String(100), nullable=False, default="text-embedding-004")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_embeddings_entity", "entity_type", "entity_id"),
    )

class PostModel(Base):
    __tablename__ = "posts"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    expected_category = Column(String(100), nullable=True)
    ground_truth_image_id = Column(String(36), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    suggestions = relationship("SuggestionModel", back_populates="post", cascade="all, delete-orphan")

class SuggestionModel(Base):
    __tablename__ = "suggestions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    post_id = Column(String(36), ForeignKey("posts.id", ondelete="CASCADE"), nullable=False, index=True)
    image_id = Column(String(36), ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True)
    rank = Column(Integer, nullable=False)
    similarity_score = Column(Float, nullable=False)
    guard_status = Column(String(20), nullable=False, index=True)  # 'APPROVED', 'REJECTED'
    rejection_reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    post = relationship("PostModel", back_populates="suggestions")
    image = relationship("ImageModel", back_populates="suggestions")
    reviews = relationship("ReviewModel", back_populates="suggestion", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_suggestions_post_image", "post_id", "image_id"),
    )

class ReviewModel(Base):
    __tablename__ = "reviews"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    suggestion_id = Column(String(36), ForeignKey("suggestions.id", ondelete="CASCADE"), nullable=False, index=True)
    post_id = Column(String(36), nullable=False, index=True)
    image_id = Column(String(36), nullable=False, index=True)
    decision = Column(String(20), nullable=False)  # 'APPROVED', 'REJECTED'
    notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    suggestion = relationship("SuggestionModel", back_populates="reviews")

class CostLogModel(Base):
    __tablename__ = "cost_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    operation = Column(String(50), nullable=False, index=True)  # 'vision_classification', 'text_embedding'
    model = Column(String(100), nullable=False)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    estimated_cost_usd = Column(Float, default=0.0)
    item_identifier = Column(String(255), nullable=True)
    timestamp = Column(DateTime(timezone=True), index=True, default=lambda: datetime.now(timezone.utc))
