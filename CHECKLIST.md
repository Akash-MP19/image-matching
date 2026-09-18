# Capstone Requirements Verification Checklist

Every requirement specified in the capstone brief has been verified through automated tests, execution scripts, and live acceptance probes.

---

## 1. Core Requirements (Section 6)

| # | Requirement | Status | Evidence / Verification Location | Details |
|---|---|---|---|---|
| 1 | **Structured Vision Output**<br>Vision model produces structured output validated against a schema; invalid responses are never trusted. | **PASS** | `src/schemas/vision.py`<br>`tests/test_schema_validation.py` | Validated by Pydantic V2 model `VisionMetadata`. Verified in 5 unit tests rejecting empty strings, invalid confidence (>1.0, <0.0), and missing fields. |
| 2 | **Low-Confidence Flagging**<br>Low-confidence classifications are flagged instead of accepted. | **PASS** | `src/services/vision_service.py`<br>`PROBE 1` | `blurry_mystery_01.jpg` (conf: 0.42) and `abstract_noise_02.jpg` (conf: 0.38) flagged as `flagged_low_confidence`. |
| 3 | **Batch Background Processing**<br>Images are processed through a batch background job with retries. | **PASS** | `src/services/batch_processor.py`<br>`POST /api/v1/batch/ingest` | Asynchronous worker with exponential backoff retries (`MAX_BATCH_RETRIES = 3`) and status tracking. |
| 4 | **Cost Tracking**<br>Vision and embedding costs are tracked per call. | **PASS** | `src/services/cost_tracker.py`<br>`GET /api/v1/costs/ledger`<br>`PROBE 6` | Per-call attribution logging tokens, model, operation, and estimated USD with budget guard ($5.00 limit). |
| 5 | **Embeddings & Ranking**<br>Image and post embeddings are stored; posts return ranked image suggestions. | **PASS** | `src/services/embedding_service.py`<br>`GET /api/v1/posts/{id}/images`<br>`PROBE 2` | 256-dimensional semantic embeddings stored in database; ranked by cosine similarity descending. |
| 6 | **Semantic Concept Matching**<br>Semantic matching works for equivalent concepts — "red fox" matches "Vulpes vulpes". | **PASS** | `tests/test_similarity_ranking.py`<br>`eval/evaluate.py` | Cosine similarity between "red fox" and "Vulpes vulpes" exceeds 0.60; positive match in evaluation benchmark. |
| 7 | **Mismatch Guard Rejection**<br>The mismatch guard rejects incorrect recommendations — the wolf-on-a-fox-post scenario provably fails. | **PASS** | `src/guard/mismatch_guard.py`<br>`POST /api/v1/posts/{id}/force-candidate`<br>`PROBE 3` | Guard evaluates category consistency. Forcing wolf on fox post fails with `guard_status: "REJECTED"`. |
| 8 | **Human-Readable Rejection Explanations**<br>Rejections include a human-readable explanation. | **PASS** | `src/guard/mismatch_guard.py`<br>`PROBE 3`, `PROBE 4` | Rejection reason: `"Animal category mismatch: expected fox, detected wolf"`. |
| 9 | **"No Confident Match" Behavior**<br>When no image clears the bar, the system answers "no confident match" with reasons. | **PASS** | `src/guard/mismatch_guard.py`<br>`GET /api/v1/posts/{id}/images`<br>`PROBE 4` | Unmatched posts (e.g. quantum computing) return `status: "NO_CONFIDENT_MATCH"` with itemized candidate failure reasons. |
| 10 | **Database Models & Indexes**<br>Database models for images, tags, embeddings, posts, suggestions, approvals/rejections — with required indexes. | **PASS** | `src/models/entities.py`<br>`alembic/versions/001_initial_schema.py` | Models for `images`, `embeddings`, `posts`, `suggestions`, `reviews`, `cost_logs` with indexes on category, status, entity, post/image, and timestamp. |
| 11 | **Validated API & Review Workflow**<br>API endpoints validated; the review workflow (approve / reject / inspect why) exists. | **PASS** | `src/api/v1/review.py`<br>`src/api/v1/posts.py`<br>`tests/test_review_api.py` | Human-in-the-loop endpoints (`/review/suggestions`, `/review/suggestions/{id}`) allowing approval/rejection and decision inspection. |
| 12 | **Top-1 Precision Evaluation**<br>A small labeled evaluation dataset measures top-1 precision — the number is in your README. | **PASS** | `data/eval_posts.json`<br>`eval/evaluate.py`<br>`PROBE 5`<br>`README.md` | Labeled benchmark across 12 posts (10 positive + 2 negative controls) measures **100.0% Top-1 Precision**, documented in README. |
| 13 | **Documentation Pack**<br>README with architecture explanation and diagram; the required files from Section 11 present. | **PASS** | `README.md`, `capstone.yaml`, `EVIDENCE.md`, `BUILDLOG.md`, `.env.example` | Complete submission pack present and verified. |

---

## 2. Shared Requirements

| # | Requirement | Status | Evidence / Verification Location | Details |
|---|---|---|---|---|
| 1 | **Layered Architecture** | **PASS** | `src/models/`, `src/services/`, `src/guard/`, `src/api/` | Clear decoupling: Database models $\rightarrow$ Domain services & guard $\rightarrow$ HTTP REST API routers. |
| 2 | **Validation at Boundary** | **PASS** | `src/main.py`<br>`tests/test_review_api.py::test_boundary_validation_returns_422` | Request validation handler transforms invalid inputs into clean HTTP 422 JSON errors; never returns unhandled 500s. |
| 3 | **Background Job Discipline** | **PASS** | `src/services/batch_processor.py`<br>`POST /api/v1/batch/ingest` | Heavy AI ingestion runs asynchronously with progress tracking and failure logging. |
| 4 | **Real Persistence & Migrations** | **PASS** | `alembic/versions/001_initial_schema.py`<br>`docker-compose.yml` | PostgreSQL 16 schema managed via Alembic migrations with proper index structures. |
| 5 | **Idempotency** | **PASS** | `src/services/batch_processor.py`<br>`tests/test_batch_processing.py::test_batch_processor_idempotency` | Duplicate ingestion runs detect existing filenames in DB and skip without creating duplicate rows. |
| 6 | **Secrets Clean** | **PASS** | `.env.example`<br>`.gitignore` | Zero credentials committed. Environment variables loaded via `src/config.py`. |
| 7 | **Cost Tracked & Budget Guard** | **PASS** | `src/services/cost_tracker.py`<br>`tests/test_cost_tracking.py::test_budget_limit_guard` | Every call attributed in ledger; raises `BudgetExceededError` if spend exceeds $5.00 limit. |

---

## 3. Acceptance Probes (Behavioral Pass/Fail)

| Probe | Description | Result | Output Snippet |
|---|---|---|---|
| **PROBE 1** | Batch job tags corpus; low-confidence image flagged | **PASS** | `48 processed, 2 flagged (abstract_noise_02.jpg, blurry_mystery_01.jpg)` |
| **PROBE 2** | Query 'red fox' article $\rightarrow$ fox ranks #1; wolf and dog rank lower | **PASS** | `fox_06.jpg #1 (Sim: 0.5682); wolves and dogs ranked significantly lower (< 0.20)` |
| **PROBE 3** | Force wolf candidate on fox post $\rightarrow$ rejected with explanation | **PASS** | `Guard Decision: REJECTED | Reason: "Animal category mismatch: expected fox, detected wolf"` |
| **PROBE 4** | Query post with no suitable image $\rightarrow$ 'no confident match' + reasons | **PASS** | `Match Status: NO_CONFIDENT_MATCH | Itemized reasons: similarity below 0.55 threshold` |
| **PROBE 5** | Run eval script $\rightarrow$ top-1 precision reported matching README | **PASS** | `Total Posts: 12 | Top-1 Matches: 10/10 | Safe Rejections: 2/2 | Top-1 Precision: 100.0%` |
| **PROBE 6** | Check cost log $\rightarrow$ every vision/embedding call attributed | **PASS** | `102 calls recorded | Total spend: $0.001201 | Operations: vision_classification, text_embedding` |
