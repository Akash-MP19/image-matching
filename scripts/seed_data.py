import sys
import asyncio
from pathlib import Path
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import AsyncSessionLocal, init_db
from src.models.entities import PostModel, ImageModel
from src.services.batch_processor import BatchProcessor
from src.services.embedding_service import EmbeddingService
from src.services.cost_tracker import CostTracker
from src.config import settings

SAMPLE_POSTS = [
    {
        "id": "post_fox_01",
        "title": "The behavior of red foxes",
        "content": "A comprehensive study on the foraging habits, social structure, and seasonal adaptation of wild red foxes living in temperate forest regions.",
        "expected_category": "fox"
    },
    {
        "id": "post_wolf_01",
        "title": "Hunting Habits of the Gray Wolf",
        "content": "Understanding the complex hunting tactics, vocal communication, and alpha hierarchy within gray wolf packs in snowy wilderness environments.",
        "expected_category": "wolf"
    },
    {
        "id": "post_dog_01",
        "title": "Domestic Dog Behavior and Training",
        "content": "Guidelines for positive reinforcement, companionship, and daily exercise routines for domestic dogs and household pets.",
        "expected_category": "dog"
    },
    {
        "id": "post_bear_01",
        "title": "Grizzly Bears of the Pacific Northwest",
        "content": "Tracking coastal brown grizzly bears as they fish for migrating salmon during late summer along rushing mountain rivers.",
        "expected_category": "bear"
    },
    {
        "id": "post_deer_01",
        "title": "White-Tailed Deer in Early Morning Meadows",
        "content": "Observations of white-tailed stags and does grazing quietly among misty woodland clearings and clover meadows.",
        "expected_category": "deer"
    },
    {
        "id": "post_quantum_01",
        "title": "Quantum Supremacy and Cryogenic Circuits",
        "content": "A deep dive into superconducting transmon qubits, Josephson junctions, and microwave control lines in quantum computing laboratories.",
        "expected_category": None
    }
]

async def seed():
    print("=" * 60)
    print("Seeding Image Matching Engine Database")
    print("=" * 60)

    # 1. Initialize schema
    await init_db()

    async with AsyncSessionLocal() as session:
        # 2. Ingest corpus
        print("[1/3] Processing image corpus...")
        batch_res = await BatchProcessor.process_corpus(session)
        print(f"      Images processed: {batch_res['processed_count']} (Flagged low-confidence: {batch_res['flagged_count']})")

        # 3. Seed posts
        print("[2/3] Seeding demo blog posts...")
        for p in SAMPLE_POSTS:
            existing = (await session.execute(select(PostModel).where(PostModel.id == p["id"]))).scalar_one_or_none()
            if not existing:
                new_post = PostModel(
                    id=p["id"],
                    title=p["title"],
                    content=p["content"],
                    expected_category=p["expected_category"]
                )
                session.add(new_post)
                await session.flush()

                # Generate embedding
                combined = f"{p['title']}. {p['content']}"
                vec = await EmbeddingService.get_embedding(combined)
                await CostTracker.record_cost(
                    session=session,
                    operation="text_embedding",
                    model=settings.EMBEDDING_MODEL,
                    input_tokens=len(combined.split()),
                    item_identifier=f"post_{p['id']}"
                )
                from src.models.entities import EmbeddingModel
                emb = EmbeddingModel(
                    entity_type="post",
                    entity_id=new_post.id,
                    vector=vec,
                    dimension=len(vec),
                    model=settings.EMBEDDING_MODEL
                )
                session.add(emb)
                print(f"      Seeded post: '{p['title']}'")

        await session.commit()

        # 4. Summary
        img_count = len((await session.execute(select(ImageModel))).scalars().all())
        post_count = len((await session.execute(select(PostModel))).scalars().all())
        cost_summary = await CostTracker.get_summary(session)

        print("[3/3] Seeding complete!")
        print(f"      Total Images in DB: {img_count}")
        print(f"      Total Posts in DB:  {post_count}")
        print(f"      Total AI Spend:     ${cost_summary['total_cost_usd']:.6f} ({cost_summary['total_calls']} calls)")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(seed())
