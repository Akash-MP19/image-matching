import pytest
from src.guard.mismatch_guard import MismatchGuard
from src.models.entities import ImageModel, PostModel

def test_guard_rejects_wolf_on_fox_post():
    post_title = "The behavior of red foxes"
    post_content = "A study of wild red foxes foraging in autumn forests."
    candidate = ImageModel(
        id="test_wolf",
        filename="wolf_01.jpg",
        filepath="/data/wolf_01.jpg",
        category="animal",
        subject="gray wolf",
        attributes=["gray fur", "pack predator"],
        caption="A gray wolf in the snowy forest",
        confidence=0.95,
        status="processed"
    )

    is_approved, reason = MismatchGuard.evaluate_candidate(
        post_title=post_title,
        post_content=post_content,
        candidate_image=candidate,
        similarity_score=0.78,
        post_expected_category="fox"
    )

    assert not is_approved
    assert "Animal category mismatch" in reason
    assert "expected fox" in reason
    assert "detected wolf" in reason

def test_guard_approves_valid_fox_match():
    post_title = "The behavior of red foxes"
    post_content = "A study of wild red foxes foraging in autumn forests."
    candidate = ImageModel(
        id="test_fox",
        filename="fox_01.jpg",
        filepath="/data/fox_01.jpg",
        category="animal",
        subject="red fox",
        attributes=["orange fur", "bushy tail"],
        caption="A vibrant red fox in an autumn forest",
        confidence=0.94,
        status="processed"
    )

    is_approved, reason = MismatchGuard.evaluate_candidate(
        post_title=post_title,
        post_content=post_content,
        candidate_image=candidate,
        similarity_score=0.82,
        post_expected_category="fox"
    )

    assert is_approved
    assert "Approved" in reason

def test_guard_rejects_low_similarity():
    post_title = "The behavior of red foxes"
    post_content = "A study of wild red foxes."
    candidate = ImageModel(
        id="test_fox",
        filename="fox_01.jpg",
        filepath="/data/fox_01.jpg",
        category="animal",
        subject="red fox",
        attributes=["orange fur"],
        caption="A red fox",
        confidence=0.94,
        status="processed"
    )

    is_approved, reason = MismatchGuard.evaluate_candidate(
        post_title=post_title,
        post_content=post_content,
        candidate_image=candidate,
        similarity_score=0.35,  # below threshold 0.55
        post_expected_category="fox"
    )

    assert not is_approved
    assert "below required confidence threshold" in reason

def test_guard_rejects_low_confidence_flagged_image():
    post_title = "The behavior of red foxes"
    post_content = "A study of wild red foxes."
    candidate = ImageModel(
        id="test_blurry",
        filename="blurry_01.jpg",
        filepath="/data/blurry_01.jpg",
        category="animal",
        subject="red fox",
        attributes=["blurry"],
        caption="A very blurry reddish shape",
        confidence=0.45,
        status="flagged_low_confidence",
        flag_reason="Confidence below 0.70"
    )

    is_approved, reason = MismatchGuard.evaluate_candidate(
        post_title=post_title,
        post_content=post_content,
        candidate_image=candidate,
        similarity_score=0.85,
        post_expected_category="fox"
    )

    assert not is_approved
    assert "low model confidence" in reason
