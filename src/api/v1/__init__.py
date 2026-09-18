from fastapi import APIRouter

from src.api.v1.images import router as images_router
from src.api.v1.posts import router as posts_router
from src.api.v1.batch import router as batch_router
from src.api.v1.review import router as review_router
from src.api.v1.costs import router as costs_router
from src.api.v1.eval import router as eval_router

v1_router = APIRouter(prefix="/api/v1")

v1_router.include_router(images_router)
v1_router.include_router(posts_router)
v1_router.include_router(batch_router)
v1_router.include_router(review_router)
v1_router.include_router(costs_router)
v1_router.include_router(eval_router)
