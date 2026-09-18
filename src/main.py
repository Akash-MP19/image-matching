import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.config import settings
from src.database import init_db
from src.api.v1 import v1_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database tables...")
    await init_db()
    yield
    logger.info("Shutting down image matching service.")

app = FastAPI(
    title="AI Image Understanding & Content Matching Engine",
    description="Automated image library understanding, structured vision tagging, semantic vector ranking, and mismatch guard safety layer.",
    version="1.0.0",
    lifespan=lifespan
)

from fastapi.encoders import jsonable_encoder

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "The provided request body or parameters failed validation at the boundary.",
            "details": jsonable_encoder(exc.errors())
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected server error occurred. Please check logs."
        }
    )

# Include API v1 router
app.include_router(v1_router)

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for container orchestrators and evaluators."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "offline_mode": settings.OFFLINE_MODE,
        "vision_model": settings.VISION_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL
    }

@app.get("/", tags=["System"])
async def root():
    """API overview and entry points."""
    return {
        "name": "AI Image Understanding & Content Matching Engine",
        "docs_url": "/docs",
        "health_url": "/health",
        "api_v1_endpoints": {
            "batch_ingest": "POST /api/v1/batch/ingest",
            "batch_status": "GET /api/v1/batch/status",
            "images": "GET /api/v1/images",
            "posts": "GET /api/v1/posts, POST /api/v1/posts",
            "match_images": "GET /api/v1/posts/{id}/images",
            "force_candidate": "POST /api/v1/posts/{id}/force-candidate",
            "reviews": "GET /api/v1/review/suggestions, POST /api/v1/review/suggestions/{id}",
            "costs": "GET /api/v1/costs, GET /api/v1/costs/ledger",
            "eval": "POST /api/v1/eval/run, GET /api/v1/eval/results"
        }
    }

@app.get("/admin", response_class=HTMLResponse, tags=["Admin Interface"])
async def minimal_admin_interface():
    """Minimal internal inspection page for reviewing suggestions and audit decisions."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AI Content Matching Engine - Admin Dashboard</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 30px; background: #0f172a; color: #f8fafc; }
            h1 { color: #38bdf8; font-size: 24px; margin-bottom: 8px; }
            p { color: #94a3b8; font-size: 14px; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; background: #1e293b; border-radius: 8px; overflow: hidden; }
            th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; font-size: 13px; }
            th { background: #334155; color: #e2e8f0; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; }
            .badge-approved { background: #10b981; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
            .badge-rejected { background: #ef4444; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
            .btn { background: #38bdf8; color: #0f172a; border: none; padding: 8px 14px; border-radius: 6px; font-weight: 600; cursor: pointer; }
            .btn:hover { background: #7dd3fc; }
            #status-bar { margin: 15px 0; padding: 12px; background: #1e293b; border-left: 4px solid #38bdf8; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>Content Matching Engine — Inspection Dashboard</h1>
        <p>Real-time review and mismatch guard decision inspector</p>
        <div id="status-bar">Loading system status...</div>
        <button class="btn" onclick="fetchSuggestions()">Refresh Suggestions</button>
        <button class="btn" onclick="runEval()">Run Evaluation</button>

        <table id="suggestions-table">
            <thead>
                <tr>
                    <th>Post Title</th>
                    <th>Candidate Image</th>
                    <th>Subject</th>
                    <th>Similarity</th>
                    <th>Guard Decision</th>
                    <th>Guard Reason / Explanation</th>
                </tr>
            </thead>
            <tbody id="table-body">
                <tr><td colspan="6" style="text-align:center; color:#64748b;">No suggestions loaded. Click Refresh.</td></tr>
            </tbody>
        </table>

        <script>
            async function fetchSuggestions() {
                try {
                    const res = await fetch('/api/v1/review/suggestions');
                    const data = await res.json();
                    const tbody = document.getElementById('table-body');
                    if (!data || data.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color:#94a3b8;">No pairing suggestions recorded yet. Ingest corpus and query a post.</td></tr>';
                        return;
                    }
                    tbody.innerHTML = data.map(s => `
                        <tr>
                            <td><strong>${s.post_title}</strong></td>
                            <td><code>${s.image_filename}</code></td>
                            <td>${s.image_subject}</td>
                            <td>${(s.similarity_score * 100).toFixed(1)}%</td>
                            <td><span class="${s.guard_status === 'APPROVED' ? 'badge-approved' : 'badge-rejected'}">${s.guard_status}</span></td>
                            <td>${s.rejection_reason || 'Verified taxonomic and semantic match'}</td>
                        </tr>
                    `).join('');
                } catch(e) {
                    alert('Error loading suggestions: ' + e);
                }
            }
            async function runEval() {
                document.getElementById('status-bar').innerText = 'Running evaluation...';
                try {
                    const res = await fetch('/api/v1/eval/run', { method: 'POST' });
                    const data = await res.json();
                    document.getElementById('status-bar').innerText = `Evaluation Completed: Top-1 Precision = ${data.precision_percentage} (${data.top1_matches + data.safe_rejections}/${data.total_posts} cases passed)`;
                } catch(e) {
                    document.getElementById('status-bar').innerText = 'Evaluation failed: ' + e;
                }
            }
            fetch('/health').then(r => r.json()).then(d => {
                document.getElementById('status-bar').innerText = `Status: ${d.status.toUpperCase()} | Mode: ${d.offline_mode ? 'Deterministic Offline' : 'Live Gemini'} | Vision Model: ${d.vision_model}`;
            });
        </script>
    </body>
    </html>
    """
