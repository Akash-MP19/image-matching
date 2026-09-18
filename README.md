# AI Image Understanding & Content Matching Engine

A production-ready, trustworthy AI decision system that ingests an image library, extracts schema-validated structured vision metadata, generates vector embeddings, ranks candidates for written articles using semantic similarity, and applies an intelligent **Mismatch Guard** safety layer to provably reject incorrect pairings with human-readable explanations.

---

## 1. System Overview

Standard semantic search engines blindly surface the highest cosine similarity vector, frequently resulting in subtle but damaging domain errors (e.g. recommending a gray wolf photo for an article about red foxes because both are wild canines in snowy forests).

This engine treats AI as an unreliable-but-useful component: **good suggestions when confident, safe rejection when uncertain**.

### Key Capabilities
- **Structured Vision Understanding**: Validates visual output against strict schemas (`subject`, `category`, `attributes`, `caption`, `confidence`).
- **Low-Confidence Flagging**: Automatically flags degraded, blurry, or ambiguous images for human review rather than silently guessing.
- **Asynchronous Batch Processing**: Background queue with exponential backoff retries and per-call AI cost tracking.
- **Taxonomic & Semantic Vector Search**: Embeds captions and articles into a shared conceptual vector space; recognizes synonyms and biological equivalents (`"red fox"` $\leftrightarrow$ `"Vulpes vulpes"`).
- **The Mismatch Guard**: Multi-stage safety layer evaluating confidence scores, similarity thresholds, and taxonomic consistency to reject incorrect candidates (e.g., wolf for fox) with explanatory feedback.
- **Human-in-the-Loop Review**: REST API and internal admin dashboard for inspecting, approving, or rejecting pairings.

---

## 2. Architecture Diagram

```
                                  +-----------------------+
                                  |  Image Corpus (48)    |
                                  +-----------+-----------+
                                              |
                                              v
                              +-------------------------------+
                              | Async Batch Processing Worker |
                              | - Retries (Exponential Backoff)|
                              | - Per-Call Cost Tracking      |
                              +---------------+---------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
        +-------------------------+                       +-------------------------+
        |  Vision Service         |                       | Embedding Engine        |
        |  - Gemini 2.0 / Offline |                       | - 256-Dim Semantic Space|
        |  - Pydantic Validation  |                       | - Concept Clustering    |
        |  - Low-Confidence Flag  |                       | - Cosine Similarity     |
        +------------+------------+                       +------------+------------+
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                              v
                                  +-----------------------+
                                  |  PostgreSQL Database  |
                                  |  - images & metadata  |
                                  |  - vectors & posts    |
                                  |  - suggestions & costs|
                                  +-----------+-----------+
                                              ^
                                              |
[ Blog Article / Post ] ----> [ Embed Content ]
                                              |
                                              v
                                 +-------------------------+
                                 | Top-K Candidate Ranking |
                                 +------------+------------+
                                              |
                                              v
                                 +-------------------------+
                                 |   THE MISMATCH GUARD    |
                                 | - Low-Confidence Check  |
                                 | - Taxonomic Conflict    |
                                 | - Category Alignment    |
                                 | - Similarity Threshold  |
                                 +------------+------------+
                                              |
                         +--------------------+--------------------+
                         |                                         |
                         v                                         v
              [ Status: MATCH_FOUND ]                  [ Status: NO_CONFIDENT_MATCH ]
              Suggested Candidate +                    Detailed List of Itemized
              Approval Explanation                     Rejection Reasons
                         |                                         |
                         +--------------------+--------------------+
                                              |
                                              v
                                  +-----------------------+
                                  | Review & Audit API    |
                                  | - Approve / Reject    |
                                  | - Cost Ledger Audit   |
                                  | - Admin Dashboard     |
                                  +-----------------------+
```

---

## 3. The Mismatch Guard (Safety Layer)

The Mismatch Guard evaluates retrieved image candidates before presenting them to users.

### Decision Rules (Evaluated in Precedence Order):
1. **Quality & Confidence Gate**: If an image was flagged during ingestion as low-confidence (`confidence < 0.70`) or degraded quality, it is categorically refused.
2. **Taxonomic Category Conflict Guard**: Validates that the entity discussed in the post matches the subject depicted in the image. If an article discusses a **fox**, but the candidate image depicts a **wolf**, the candidate is immediately rejected:
   > *"Animal category mismatch: expected fox, detected wolf"*
3. **Explicit Category Alignment**: Verifies that high-level domain constraints match.
4. **Similarity Threshold Gate**: Ensures the cosine similarity score clears the calibrated confidence threshold ($\ge 0.55$).
5. **No Confident Match Fallback**: When no candidate clears all rules, the system returns status `"NO_CONFIDENT_MATCH"` with itemized rejection explanations instead of guessing.

---

## 4. Evaluation Benchmark Results

The engine includes an automated benchmark evaluating 12 labeled article scenarios (10 positive retrieval tests across 5 categories + 2 negative out-of-domain controls):

- **Benchmark Command**: `python eval/evaluate.py`
- **Total Posts Evaluated**: 12
- **Positive Ground-Truth Retrieval Hits**: 10 / 10
- **Safe Mismatch Guard Rejections**: 2 / 2
- **Measured Top-1 Precision**: **100.0%** (12/12 cases passed)

---

## 5. Quick Start & Setup

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional for containerized PostgreSQL)

### 1. Installation
```bash
git clone https://github.com/Akash-MP19/image-matching.git
cd image-matching
python -m pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(By default, `OFFLINE_MODE=true` is enabled for deterministic, instant execution without API keys. To connect live Gemini Flash, set `OFFLINE_MODE=false` and provide your `GEMINI_API_KEY`.)*

### 3. Database Seeding & Corpus Ingestion
Run the seed script to initialize tables, ingest the 48-image corpus, generate embeddings, and create sample posts:
```bash
python scripts/seed_data.py
```

### 4. Running the Service
Start the FastAPI server:
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)
- Minimal Inspection Dashboard: [http://localhost:8000/admin](http://localhost:8000/admin)

---

## 6. Verification & Acceptance Probes

To run the automated test suite (22 unit & integration tests):
```bash
python -m pytest -v
```

To execute the 6 behavioral acceptance probes:
```bash
python scripts/run_probes.py
```

To execute the labeled benchmark:
```bash
python eval/evaluate.py
```

All requirement verifications are detailed in [`CHECKLIST.md`](./CHECKLIST.md) and [`EVIDENCE.md`](./EVIDENCE.md), and service entry points are defined in [`manifest.yaml`](./manifest.yaml).

### Acceptance Probes Summary:
- **Probe 1**: Ingests image corpus $\rightarrow$ 48 images processed, 2 low-confidence images cleanly flagged.
- **Probe 2**: Queries "red fox" post $\rightarrow$ fox image ranks first (0.5682); wolf and dog rank significantly lower.
- **Probe 3**: Forces wolf as candidate for fox post $\rightarrow$ rejected with `"Animal category mismatch: expected fox, detected wolf"`.
- **Probe 4**: Queries out-of-domain post $\rightarrow$ returns `"NO_CONFIDENT_MATCH"` with itemized reasons.
- **Probe 5**: Executes benchmark suite $\rightarrow$ measures Top-1 precision at 100.0%.
- **Probe 6**: Audits cost log $\rightarrow$ all calls attributed with costs, tokens, and timestamps.

---

## 7. Running with Docker Compose (PostgreSQL)

To run the complete production environment with PostgreSQL 16:
```bash
docker compose up -d --build
docker compose exec app python scripts/seed_data.py
```

---

## 8. Limitations & Future Work

1. **Static Taxonomy Dictionary**: The offline taxonomic conflict detector utilizes a curated taxonomy tree for canids, ursids, and cervids. Future iterations could use a dynamic Knowledge Graph or zero-shot NLI entailment classifier.
2. **Multi-Modal Joint Embeddings**: Captions are currently projected via text vectors. Integrating joint vision-language encoders (e.g., CLIP/SigLIP) would enable direct image-to-text embedding comparisons.
3. **Corpus Scale**: The current test library contains 48 images across 5 animal categories. At scales $> 100,000$ images, an approximate nearest neighbor (ANN) index (HNSW via pgvector or Milvus) is recommended.
