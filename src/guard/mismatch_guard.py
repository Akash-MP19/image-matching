import re
import logging
from typing import List, Tuple, Optional, Dict, Any

from src.models.entities import ImageModel, PostModel
from src.schemas.suggestion import SuggestedCandidate, PostMatchResponse
from src.config import settings

logger = logging.getLogger(__name__)

# Canonical taxonomic entities
ENTITY_KEYWORDS = {
    "fox": ["fox", "foxes", "red fox", "vulpes", "vulpes vulpes", "vulpine", "kit fox"],
    "wolf": ["wolf", "wolves", "gray wolf", "grey wolf", "canis lupus", "timber wolf", "lupine"],
    "dog": ["dog", "dogs", "puppy", "canis familiaris", "domestic dog", "canine pet", "golden retriever"],
    "bear": ["bear", "bears", "grizzly", "brown bear", "ursus", "ursus arctos", "black bear"],
    "deer": ["deer", "deers", "stag", "doe", "fawn", "cervidae", "white-tailed deer", "elk"]
}

class MismatchGuard:
    @staticmethod
    def detect_entity(text: str) -> Optional[str]:
        """Detects primary animal entity mentioned in text."""
        text_lower = text.lower()
        for entity, keywords in ENTITY_KEYWORDS.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    return entity
        return None

    @classmethod
    def evaluate_candidate(
        cls,
        post_title: str,
        post_content: str,
        candidate_image: ImageModel,
        similarity_score: float,
        post_expected_category: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Production AI Safety Layer:
        Evaluates whether a candidate image is trustworthy and safe to recommend.
        
        Returns:
            (is_approved: bool, reason_or_explanation: str)
        """
        # Rule 1: Flagged Image Quality Gate
        if candidate_image.status == "flagged_low_confidence":
            reason = (
                f"Candidate rejected: Image has low model confidence ({candidate_image.confidence:.2f}) "
                f"or ambiguous quality ({candidate_image.flag_reason or 'unclear subject'})."
            )
            return False, reason

        # Rule 2: Taxonomic / Category Mismatch Guard (Semantic Safety Layer)
        # Determine target entity from post content, title, and expected_category
        post_text = f"{post_title} {post_content} {post_expected_category or ''}"
        post_entity = cls.detect_entity(post_text) or (post_expected_category.lower() if post_expected_category else None)
        
        candidate_text = f"{candidate_image.subject} {candidate_image.category} {' '.join(candidate_image.attributes or [])}"
        image_entity = cls.detect_entity(candidate_text) or candidate_image.subject.lower()

        if post_entity and image_entity:
            # Check for known taxonomic conflicts (e.g. fox vs wolf, fox vs dog)
            if post_entity != image_entity:
                reason = f"Animal category mismatch: expected {post_entity}, detected {image_entity}"
                return False, reason

        # Rule 3: Explicit Subject / Category Discrepancy
        if post_expected_category and post_expected_category.lower() not in candidate_image.category.lower() and post_expected_category.lower() not in candidate_image.subject.lower():
            reason = f"Category mismatch: expected '{post_expected_category}', detected '{candidate_image.category}'"
            return False, reason

        # Rule 4: Minimum Semantic Similarity Threshold
        if similarity_score < settings.SIMILARITY_THRESHOLD:
            reason = (
                f"Candidate rejected: Semantic similarity score ({similarity_score:.4f}) "
                f"is below required confidence threshold ({settings.SIMILARITY_THRESHOLD:.2f})."
            )
            return False, reason

        # If all safety rules pass:
        explanation = (
            f"Approved: High semantic relevance ({similarity_score:.4f}) and validated entity match "
            f"('{candidate_image.subject}')."
        )
        return True, explanation

    @classmethod
    def filter_and_rank(
        cls,
        post: PostModel,
        scored_candidates: List[Tuple[ImageModel, float]]
    ) -> PostMatchResponse:
        """
        Filters and evaluates ranked candidates for a post through the Mismatch Guard.
        Returns a structured PostMatchResponse with decisions, rankings, and explanations.
        """
        evaluated_candidates: List[SuggestedCandidate] = []
        rejection_reasons: List[str] = []
        approved_candidate: Optional[SuggestedCandidate] = None

        for rank, (image, score) in enumerate(scored_candidates, start=1):
            is_approved, explanation = cls.evaluate_candidate(
                post_title=post.title,
                post_content=post.content,
                candidate_image=image,
                similarity_score=score,
                post_expected_category=post.expected_category
            )

            candidate_dto = SuggestedCandidate(
                image_id=image.id,
                filename=image.filename,
                category=image.category,
                subject=image.subject,
                caption=image.caption,
                attributes=image.attributes or [],
                similarity_score=score,
                guard_status="APPROVED" if is_approved else "REJECTED",
                rejection_reason=explanation if not is_approved else None,
                explanation=explanation if is_approved else None
            )
            evaluated_candidates.append(candidate_dto)

            if not is_approved:
                rejection_reasons.append(f"Rank {rank} ({image.filename}): {explanation}")
            elif approved_candidate is None:
                # Top-ranked approved candidate
                approved_candidate = candidate_dto

        if approved_candidate:
            status = "MATCH_FOUND"
            decision_summary = (
                f"Candidate '{approved_candidate.filename}' approved at Rank 1 with similarity "
                f"{approved_candidate.similarity_score:.4f}."
            )
        else:
            status = "NO_CONFIDENT_MATCH"
            decision_summary = (
                "No confident match: all candidates failed the mismatch guard or fell below "
                f"the similarity threshold ({settings.SIMILARITY_THRESHOLD:.2f})."
            )

        return PostMatchResponse(
            post_id=post.id,
            post_title=post.title,
            status=status,
            suggested_image=approved_candidate,
            candidates=evaluated_candidates,
            rejection_reasons=rejection_reasons,
            decision_summary=decision_summary
        )
