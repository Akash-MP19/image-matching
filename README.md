# AI Image Understanding & Content Matching Engine

An AI-powered system that understands images and matches the right image to an article while preventing incorrect matches—such as using a wolf image for a fox article.

---

## Features

- **Structured Vision Metadata**: Extracts validated JSON metadata (`subject`, `category`, `attributes`, `caption`, `confidence`).
- **Low-Confidence Detection**: Automatically flags blurry, corrupted, or ambiguous images for human review.
- **Semantic Matching**: Embeds post text and image captions into a shared vector space for conceptual retrieval.
- **The Mismatch Guard**: Prevents false matches by checking taxonomic consistency and similarity thresholds.
- **"No Confident Match" Fallback**: Gracefully rejects out-of-domain posts with itemized reasons instead of guessing.
- **Background Batch Processing**: Asynchronous ingestion worker with exponential backoff retries.
- **Human-in-the-Loop Review**: REST API and minimal admin dashboard to inspect, approve, or reject pairings.
- **AI Cost Tracking**: Records token usage, operation costs, and enforces a hard budget guard.
- **Reproducible Modes**: Supports live Gemini 2.0 Flash (`gemini-2.0-flash`) and deterministic offline mode.

---

## Tech Stack

- **Language & Framework**: Python, FastAPI, Uvicorn
- **Database & ORM**: PostgreSQL, SQLAlchemy, Alembic
- **AI & ML**: Gemini 2.0 Flash, Semantic Text Embeddings, NumPy, Scikit-learn
- **Validation**: Pydantic V2
- **Testing & Containerization**: Pytest, Pytest-AsyncIO, Docker, Docker Compose

---

## How It Works

```
Image → Vision AI → Metadata → Embeddings → Similarity Ranking → Mismatch Guard → Match / No Confident Match
```

1. **Ingest**: Images in the 48-image library are analyzed by the vision model.
2. **Validate**: Metadata is schema-validated; low-confidence images (< 0.70) are flagged.
3. **Embed**: Image captions and article texts are mapped into dense semantic vectors.
4. **Rank**: Candidates are ranked by cosine similarity.
5. **Guard**: The Mismatch Guard verifies taxonomic consistency (e.g., rejecting a wolf candidate for a fox post) and confidence thresholds.
6. **Decide**: Returns `MATCH_FOUND` with the approved image or `NO_CONFIDENT_MATCH` with explanations.

---

## Run Locally

### 1. Clone and Install
```bash
git clone https://github.com/Akash-MP19/image-matching.git
cd image-matching
python -m pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
```
*(By default, `OFFLINE_MODE=true` is enabled for instant execution without API keys. To use live Gemini, set `GEMINI_API_KEY` in `.env`.)*

### 3. Seed Database & Ingest Corpus
Initializes database tables, ingests the 48 images, generates embeddings, and seeds demo posts:
```bash
python scripts/seed_data.py
```

### 4. Start Server
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

- **Interactive API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Admin Review Dashboard**: [http://localhost:8000/admin](http://localhost:8000/admin)
- **Service Configuration**: Entry points defined in [`manifest.yaml`](./manifest.yaml) / [`capstone.yaml`](./capstone.yaml).

---

## Testing & Verification

Run the test suite and verification scripts:
```bash
# Run 22 unit & integration tests
python -m pytest -v

# Run 6 behavioral acceptance probes
python scripts/run_probes.py

# Run labeled evaluation benchmark
python eval/evaluate.py
```

### Verification Results
- **Pytest Suite**: 22/22 tests passing (100%)
- **Acceptance Probes**: 6/6 probes passing (100%)
- **Positive Retrieval Cases**: 10/10 matched ground truth (100.0% Top-1 Precision)
- **Safe Rejection Cases**: 2/2 out-of-domain posts correctly rejected with explanation
- **Overall Benchmark Accuracy**: 12/12 cases passed (100.0%)

Detailed proofs are recorded in [`EVIDENCE.md`](./EVIDENCE.md) and [`CHECKLIST.md`](./CHECKLIST.md).

---

## Limitations

- **Dataset Scale**: Current dataset contains 48 images across 5 animal categories (`fox`, `wolf`, `dog`, `bear`, `deer`) plus edge cases.
- **Offline Taxonomy**: Deterministic offline mode uses a curated taxonomy cluster dictionary for animal species.
- **Vector Indexing**: Uses exact in-memory/DB cosine similarity suitable for small-to-medium corpora; larger production datasets (>100k) would benefit from ANN indexes (e.g. pgvector HNSW).

---

## Author

**Akash MP**  
GitHub: [https://github.com/Akash-MP19](https://github.com/Akash-MP19)
