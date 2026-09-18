import re
import math
import hashlib
import logging
from typing import List, Optional
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)

# Canonical taxonomic clusters
TAXONOMY_CLUSTERS = {
    "fox": {
        "keywords": [
            "fox", "foxes", "red fox", "vulpes", "vulpes vulpes", "wild fox", "kit fox",
            "fennec", "vulpine", "orange fur", "bushy tail"
        ],
        "cluster_id": 0
    },
    "wolf": {
        "keywords": [
            "wolf", "wolves", "gray wolf", "grey wolf", "canis lupus", "timber wolf",
            "pack predator", "howling", "alpha wolf", "lupine"
        ],
        "cluster_id": 1
    },
    "dog": {
        "keywords": [
            "dog", "dogs", "puppy", "canis familiaris", "domestic dog", "pet canine",
            "golden retriever", "shepherd", "leash", "hound"
        ],
        "cluster_id": 2
    },
    "bear": {
        "keywords": [
            "bear", "bears", "grizzly", "brown bear", "ursus", "ursus arctos",
            "black bear", "polar bear", "cub", "cubs", "salmon"
        ],
        "cluster_id": 3
    },
    "deer": {
        "keywords": [
            "deer", "deers", "stag", "doe", "fawn", "cervidae", "antlers",
            "white-tailed", "white-tailed deer", "elk", "buck"
        ],
        "cluster_id": 4
    }
}

EMBEDDING_DIM = 256

# Stop words to ignore during token projection
STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "with", "from", "by",
    "and", "or", "but", "is", "are", "was", "were", "of", "it", "its", "this", "that"
}

class EmbeddingService:
    @staticmethod
    def _deterministic_semantic_vector(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
        """
        Generates a dense, normalized semantic embedding vector.
        Maps conceptual taxonomic clusters (Fox, Wolf, Dog, Bear, Deer) to primary
        dimensions to preserve taxonomic boundaries, while projecting specific content
        keywords and n-grams across high-dimensional space for fine-grained ranking.
        """
        clean_text = text.lower()
        vec = np.zeros(dim, dtype=np.float32)

        # 1. Broad taxonomic cluster activations (dims 0 to 49: 10 dims per animal)
        for name, data in TAXONOMY_CLUSTERS.items():
            cid = data["cluster_id"]
            start_idx = cid * 10
            matched = False
            for kw in data["keywords"]:
                if re.search(r'\b' + re.escape(kw) + r'\b', clean_text):
                    vec[start_idx: start_idx + 10] += 3.0
                    matched = True
                    break
            if not matched:
                # Partial root matches
                for kw in data["keywords"]:
                    root = kw.split()[0]
                    if len(root) > 3 and root in clean_text:
                        vec[start_idx: start_idx + 10] += 1.2
                        break

        # 2. Specific keyword and n-gram hash projection (dims 50 to dim-1)
        words = [w for w in re.findall(r'\b[a-z]{3,}\b', clean_text) if w not in STOP_WORDS]
        vocab_space = dim - 50

        # Unigrams
        for w in words:
            h = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16)
            idx = 50 + (h % vocab_space)
            sign = 1.0 if ((h >> 4) % 2 == 0) else -1.0
            vec[idx] += sign * 1.5

        # Bigrams for phrases (e.g. "autumn forest", "river salmon", "rocky cliff")
        for i in range(len(words) - 1):
            bg = f"{words[i]}_{words[i+1]}"
            h = int(hashlib.sha256(bg.encode("utf-8")).hexdigest(), 16)
            idx = 50 + (h % vocab_space)
            sign = 1.0 if ((h >> 4) % 2 == 0) else -1.0
            vec[idx] += sign * 2.5

        # 3. L2 normalize vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return [round(float(x), 6) for x in vec]

    @staticmethod
    async def get_embedding(text: str) -> List[float]:
        """
        Retrieves embedding using live Gemini API if configured and online,
        otherwise uses the deterministic semantic vector generator.
        """
        if not settings.OFFLINE_MODE and settings.GEMINI_API_KEY:
            try:
                from google import genai
                client = genai.Client(api_key=settings.GEMINI_API_KEY)
                response = client.models.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    contents=text
                )
                embedding = response.embeddings[0].values
                vec = np.array(embedding, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                return [round(float(x), 6) for x in vec]
            except Exception as e:
                logger.warning(f"Live embedding API call failed: {e}. Falling back to deterministic embedding.")

        return EmbeddingService._deterministic_semantic_vector(text)

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """
        Computes cosine similarity between two vectors.
        """
        if not vec_a or not vec_b:
            return 0.0
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        dot = np.dot(a, b)
        sim = dot / (norm_a * norm_b)
        return round(float(sim), 4)
