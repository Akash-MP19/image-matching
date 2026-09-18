import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_api_health(client: AsyncClient):
    res = await client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"

@pytest.mark.asyncio
async def test_post_creation_and_matching(client: AsyncClient):
    # 1. Create post
    post_payload = {
        "title": "Autumn Forest Behaviors of Red Foxes",
        "content": "Observing wild red foxes as they hunt and stand alert in autumn forest clearings among fallen leaves.",
        "expected_category": "fox"
    }
    create_res = await client.post("/api/v1/posts", json=post_payload)
    assert create_res.status_code == 201
    post_id = create_res.json()["id"]

    # 2. Query matching images
    match_res = await client.get(f"/api/v1/posts/{post_id}/images")
    assert match_res.status_code == 200
    match_data = match_res.json()

    assert match_data["status"] in ("MATCH_FOUND", "NO_CONFIDENT_MATCH")
    assert len(match_data["candidates"]) > 0
    top_cand = match_data["candidates"][0]
    assert top_cand["suggestion_id"] is not None

    # 3. Review suggestion (Approve)
    sug_id = top_cand["suggestion_id"]
    review_res = await client.post(
        f"/api/v1/review/suggestions/{sug_id}",
        json={"decision": "APPROVED", "notes": "Verified high quality match"}
    )
    assert review_res.status_code == 201
    assert review_res.json()["decision"] == "APPROVED"

@pytest.mark.asyncio
async def test_force_candidate_evaluation_rejects_wolf(client: AsyncClient):
    # 1. Create fox post
    post_res = await client.post("/api/v1/posts", json={
        "title": "The behavior of red foxes",
        "content": "A detailed study of wild red foxes.",
        "expected_category": "fox"
    })
    post_id = post_res.json()["id"]

    # 2. Force wolf candidate
    force_res = await client.post(
        f"/api/v1/posts/{post_id}/force-candidate",
        json={"image_identifier": "wolf_01.jpg"}
    )
    assert force_res.status_code == 200
    candidate = force_res.json()
    assert candidate["guard_status"] == "REJECTED"
    assert "Animal category mismatch" in candidate["rejection_reason"]

@pytest.mark.asyncio
async def test_boundary_validation_returns_422(client: AsyncClient):
    # Invalid empty title
    res = await client.post("/api/v1/posts", json={"title": "", "content": ""})
    assert res.status_code == 422
    assert "Validation Error" in res.json()["error"]
