import pytest
from pydantic import ValidationError
from src.schemas.vision import VisionMetadata

def test_valid_vision_metadata():
    data = {
        "subject": "red fox",
        "category": "animal",
        "attributes": ["orange fur", "wild", "forest"],
        "caption": "A red fox standing in a forest",
        "confidence": 0.94
    }
    metadata = VisionMetadata.model_validate(data)
    assert metadata.subject == "red fox"
    assert metadata.confidence == 0.94
    assert len(metadata.attributes) == 3

def test_invalid_confidence_above_one():
    data = {
        "subject": "red fox",
        "category": "animal",
        "attributes": ["wild"],
        "caption": "A red fox",
        "confidence": 1.45
    }
    with pytest.raises(ValidationError):
        VisionMetadata.model_validate(data)

def test_invalid_confidence_negative():
    data = {
        "subject": "red fox",
        "category": "animal",
        "attributes": ["wild"],
        "caption": "A red fox",
        "confidence": -0.1
    }
    with pytest.raises(ValidationError):
        VisionMetadata.model_validate(data)

def test_missing_required_field():
    data = {
        "subject": "red fox",
        "category": "animal",
        "confidence": 0.9
        # missing caption
    }
    with pytest.raises(ValidationError):
        VisionMetadata.model_validate(data)

def test_empty_string_rejection():
    data = {
        "subject": "   ",
        "category": "animal",
        "caption": "A fox",
        "confidence": 0.9
    }
    with pytest.raises(ValidationError):
        VisionMetadata.model_validate(data)
