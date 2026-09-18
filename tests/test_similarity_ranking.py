import pytest
from src.services.embedding_service import EmbeddingService

def test_cosine_similarity_identity():
    vec = [0.6, 0.8, 0.0]
    sim = EmbeddingService.cosine_similarity(vec, vec)
    assert pytest.approx(sim, 0.001) == 1.0

def test_cosine_similarity_orthogonal():
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    sim = EmbeddingService.cosine_similarity(vec_a, vec_b)
    assert pytest.approx(sim, 0.001) == 0.0

def test_semantic_equivalence_fox_and_vulpes():
    vec_fox = EmbeddingService._deterministic_semantic_vector("The behavior of red foxes in the forest")
    vec_vulpes = EmbeddingService._deterministic_semantic_vector("Vulpes vulpes foraging quietly in woodland")
    sim = EmbeddingService.cosine_similarity(vec_fox, vec_vulpes)
    # High similarity for equivalent biological concepts
    assert sim > 0.60, f"Expected high similarity between red fox and Vulpes vulpes, got {sim}"

def test_semantic_separation_fox_and_wolf():
    vec_fox = EmbeddingService._deterministic_semantic_vector("The behavior of red foxes in the forest")
    vec_wolf = EmbeddingService._deterministic_semantic_vector("A pack of gray wolves hunting in the forest")
    sim = EmbeddingService.cosine_similarity(vec_fox, vec_wolf)
    # Different animal clusters must have distinct vectors
    assert sim < 0.35, f"Expected distinct separation between fox and wolf, got {sim}"

def test_semantic_separation_fox_and_dog():
    vec_fox = EmbeddingService._deterministic_semantic_vector("The behavior of red foxes in the forest")
    vec_dog = EmbeddingService._deterministic_semantic_vector("A domestic dog puppy playing in the backyard")
    sim = EmbeddingService.cosine_similarity(vec_fox, vec_dog)
    assert sim < 0.30, f"Expected distinct separation between fox and dog, got {sim}"
