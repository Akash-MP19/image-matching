import os
import sys
import json
import asyncio
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import AsyncSessionLocal, init_db
from src.models.entities import PostModel, ImageModel, EmbeddingModel
from src.services.batch_processor import BatchProcessor
from src.services.embedding_service import EmbeddingService
from src.guard.mismatch_guard import MismatchGuard
from sqlalchemy import select

async def run_evaluation_cli():
    print("=" * 80)
    print("AI Image Understanding & Content Matching Engine — Benchmark Evaluation")
    print("=" * 80)

    # Initialize database
    await init_db()

    async with AsyncSessionLocal() as session:
        # Check if corpus ingested, if not run batch ingestion
        img_count = (await session.execute(select(ImageModel))).scalars().all()
        if len(img_count) == 0:
            print("[INFO] Corpus not ingested yet. Running batch ingestion...")
            batch_result = await BatchProcessor.process_corpus(session)
            print(f"[INFO] Ingestion completed: {batch_result['processed_count']} processed, {batch_result['flagged_count']} flagged.")

        # Load eval posts
        eval_file = PROJECT_ROOT / "data" / "eval_posts.json"
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        posts = eval_data.get("posts", [])
        total = len(posts)
        top1_hits = 0
        safe_rejections = 0

        # Load all image embeddings
        img_emb_query = select(EmbeddingModel, ImageModel).join(
            ImageModel, EmbeddingModel.entity_id == ImageModel.id
        ).where(EmbeddingModel.entity_type == "image")
        image_rows = (await session.execute(img_emb_query)).all()

        print(f"\nEvaluating {total} labeled benchmark posts against {len(image_rows)} image embeddings...\n")
        print(f"{'#':<3} | {'Post Title':<35} | {'Expected':<8} | {'Status':<18} | {'Selected Image':<16} | {'Sim':<6} | {'Verdict'}")
        print("-" * 105)

        for idx, p in enumerate(posts, start=1):
            title = p["title"]
            content = p["content"]
            expected = p.get("expected_category") or "N/A"
            gt_filename = p.get("ground_truth_filename")
            is_neg = p.get("is_negative_example", False)

            # Generate post embedding
            combined_text = f"{title}. {content}"
            post_vec = await EmbeddingService.get_embedding(combined_text)

            # Score candidates
            scored = []
            for emb, img in image_rows:
                sim = EmbeddingService.cosine_similarity(post_vec, emb.vector)
                scored.append((img, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            top_candidates = scored[:5]

            dummy_post = PostModel(
                id=f"cli_{p['id']}",
                title=title,
                content=content,
                expected_category=expected if expected != "N/A" else None
            )

            match_res = MismatchGuard.filter_and_rank(dummy_post, top_candidates)

            verdict = "FAIL"
            selected_fn = match_res.suggested_image.filename if match_res.suggested_image else "None"
            sim_str = f"{match_res.suggested_image.similarity_score:.3f}" if match_res.suggested_image else "N/A"

            if is_neg:
                if match_res.status == "NO_CONFIDENT_MATCH":
                    safe_rejections += 1
                    verdict = "PASS (Safe Rejection)"
                else:
                    verdict = "FAIL (False Positive)"
            else:
                if match_res.suggested_image and match_res.suggested_image.filename == gt_filename:
                    top1_hits += 1
                    verdict = "PASS (Top-1 Match)"
                elif match_res.suggested_image:
                    verdict = f"FAIL (Got {match_res.suggested_image.filename})"
                else:
                    verdict = "FAIL (Rejected Valid Match)"

            title_trunc = (title[:32] + "...") if len(title) > 35 else title
            print(f"{idx:<3} | {title_trunc:<35} | {expected:<8} | {match_res.status:<18} | {selected_fn:<16} | {sim_str:<6} | {verdict}")

        passed = top1_hits + safe_rejections
        precision = passed / total
        pct_str = f"{precision * 100:.1f}%"

        print("-" * 105)
        print(f"\n[BENCHMARK RESULTS]")
        print(f"Total Posts Evaluated:      {total}")
        print(f"Positive Retrieval Hits:    {top1_hits} / {total - 2}")
        print(f"Safe Guard Rejections:      {safe_rejections} / 2")
        print(f"Total Benchmark Success:    {passed} / {total}")
        print(f"Measured Top-1 Precision:   {pct_str}\n")
        print("=" * 80)

        # Output JSON result file for documentation and evidence
        summary = {
            "total_posts": total,
            "top1_matches": top1_hits,
            "safe_rejections": safe_rejections,
            "total_passed": passed,
            "precision_percentage": pct_str,
            "top1_precision": round(precision, 4)
        }
        with open(PROJECT_ROOT / "data" / "eval_results.json", "w", encoding="utf-8") as out_f:
            json.dump(summary, out_f, indent=2)

if __name__ == "__main__":
    asyncio.run(run_evaluation_cli())
