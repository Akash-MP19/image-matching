"""initial schema with required indexes

Revision ID: 001_initial
Revises: 
Create Date: 2026-09-18 11:55:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # 1. Images table
    op.create_table(
        'images',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('filepath', sa.String(length=512), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=False),
        sa.Column('subject', sa.String(length=100), nullable=False),
        sa.Column('attributes', sa.JSON(), nullable=False),
        sa.Column('caption', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('flag_reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('filename')
    )
    op.create_index('ix_images_category', 'images', ['category'], unique=False)
    op.create_index('ix_images_subject', 'images', ['subject'], unique=False)
    op.create_index('ix_images_status', 'images', ['status'], unique=False)
    op.create_index('ix_images_category_subject', 'images', ['category', 'subject'], unique=False)

    # 2. Embeddings table
    op.create_table(
        'embeddings',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('entity_type', sa.String(length=20), nullable=False),
        sa.Column('entity_id', sa.String(length=36), nullable=False),
        sa.Column('vector', sa.JSON(), nullable=False),
        sa.Column('dimension', sa.Integer(), nullable=False),
        sa.Column('model', sa.String(length=100), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_embeddings_entity', 'embeddings', ['entity_type', 'entity_id'], unique=False)

    # 3. Posts table
    op.create_table(
        'posts',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('expected_category', sa.String(length=100), nullable=True),
        sa.Column('ground_truth_image_id', sa.String(length=36), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # 4. Suggestions table
    op.create_table(
        'suggestions',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('post_id', sa.String(length=36), nullable=False),
        sa.Column('image_id', sa.String(length=36), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=False),
        sa.Column('similarity_score', sa.Float(), nullable=False),
        sa.Column('guard_status', sa.String(length=20), nullable=False),
        sa.Column('rejection_reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['image_id'], ['images.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['post_id'], ['posts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_suggestions_post_image', 'suggestions', ['post_id', 'image_id'], unique=False)
    op.create_index('ix_suggestions_guard_status', 'suggestions', ['guard_status'], unique=False)

    # 5. Reviews table
    op.create_table(
        'reviews',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('suggestion_id', sa.String(length=36), nullable=False),
        sa.Column('post_id', sa.String(length=36), nullable=False),
        sa.Column('image_id', sa.String(length=36), nullable=False),
        sa.Column('decision', sa.String(length=20), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['suggestion_id'], ['suggestions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # 6. Cost Logs table
    op.create_table(
        'cost_logs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('operation', sa.String(length=50), nullable=False),
        sa.Column('model', sa.String(length=100), nullable=False),
        sa.Column('input_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),
        sa.Column('estimated_cost_usd', sa.Float(), nullable=True),
        sa.Column('item_identifier', sa.String(length=255), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_cost_logs_timestamp', 'cost_logs', ['timestamp'], unique=False)
    op.create_index('ix_cost_logs_operation', 'cost_logs', ['operation'], unique=False)

def downgrade() -> None:
    op.drop_table('cost_logs')
    op.drop_table('reviews')
    op.drop_table('suggestions')
    op.drop_table('posts')
    op.drop_table('embeddings')
    op.drop_table('images')
