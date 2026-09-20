import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.config import SAMPLES_DIR
from backend.data.sample_generator import generate_sample_tiles


@pytest.fixture(scope="module")
def client():
    # Ensure sample tiles exist
    generate_sample_tiles(SAMPLES_DIR)
    with TestClient(app) as c:
        yield c


def test_api_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_api_list_tiles(client):
    res = client.get("/api/v1/tiles")
    assert res.status_code == 200
    data = res.json()
    assert data["total_tiles"] >= 4
    assert len(data["tiles"]) >= 4


def test_api_composite_image(client):
    res = client.get("/api/v1/tiles/sample_forest_tile/composite?mode=rgb")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"


def test_api_spectral(client):
    res = client.post("/api/v1/spectral", json={"tile_id": "sample_forest_tile"})
    assert res.status_code == 200
    data = res.json()
    assert "ndvi_stats" in data
    assert "coverage_estimates" in data


def test_api_classify(client):
    res = client.post("/api/v1/classify", json={"tile_id": "sample_water_tile", "top_k": 3})
    assert res.status_code == 200
    data = res.json()
    assert "primary_class" in data
    assert len(data["top_predictions"]) == 3


def test_api_vlm_query(client):
    res = client.post(
        "/api/v1/vlm/query",
        json={"tile_id": "sample_cropland_tile", "question": "What is the dominant landscape?"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert len(data["answer"]) > 0


def test_api_agent_chat(client):
    res = client.post(
        "/api/v1/agent/chat",
        json={"tile_id": "sample_urban_tile", "query": "Assess urban infrastructure and vegetation."},
    )
    assert res.status_code == 200
    data = res.json()
    assert "response" in data
    assert len(data["plan"]) > 0


def test_api_analyze_multipart_upload(client):
    import io
    from PIL import Image

    # Create a synthetic in-memory RGB PNG image
    img = Image.new("RGB", (100, 100), color=(30, 80, 180))  # Water-like blue
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    res = client.post(
        "/analyze",
        files=[("files", ("test_water.png", buf.getvalue(), "image/png"))],
        data={"query": "Where are the water bodies?"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "confidence" in data
    assert "selected_task" in data
    assert "detected_modality" in data
    assert "execution_trace" in data
    assert data["detected_modality"] == "RGB_OPTICAL"
    assert data["selected_task"] == "grounding"
    assert len(data["execution_trace"]) >= 4


def test_api_analyze_sensor_integrity_rejection(client):
    import io
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(100, 200, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    res = client.post(
        "/analyze",
        files=[("files", ("test_veg.png", buf.getvalue(), "image/png"))],
        data={"query": "Calculate NDVI and multispectral indices for this area."},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["selected_task"] == "spectral_analysis"
    assert data["selected_model_or_tool"] == "SensorIntegrityGuard"
    assert "Capability Limitation" in data["answer"]


def test_api_analyze_empty_query_validation(client):
    import io
    from PIL import Image

    img = Image.new("RGB", (32, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    res = client.post(
        "/analyze",
        files=[("files", ("test.png", buf.getvalue(), "image/png"))],
        data={"query": "   "},
    )
    assert res.status_code == 400
    assert "Query string cannot be empty" in res.json()["detail"]


def test_api_upload_tile_and_persistence(client):
    import io
    from PIL import Image

    # Upload test tile
    img = Image.new("RGB", (64, 64), color=(50, 150, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    res = client.post(
        "/api/v1/upload-tile",
        files={"file": ("api_test_tile.png", buf.getvalue(), "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["tile_id"] == "api_test_tile"
    assert "session_id" in data
    assert data["modality"] == "RGB_OPTICAL"

    # Verify uploaded tile persists and can be queried via agent chat
    chat_res = client.post(
        "/api/v1/agent/chat",
        json={"tile_id": "api_test_tile", "query": "Describe the scene.", "session_id": data["session_id"]},
    )
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert chat_data["session_id"] == data["session_id"]
    assert chat_data["history_length"] == 2


def test_api_session_history_and_deletion(client):
    sess_id = "test_api_session_xyz"

    # Step 1: Send query with session ID
    res = client.post(
        "/api/v1/agent/chat",
        json={"tile_id": "sample_forest_tile", "query": "What is the primary land cover?", "session_id": sess_id},
    )
    assert res.status_code == 200
    assert res.json()["session_id"] == sess_id

    # Step 2: Retrieve session history
    hist_res = client.get(f"/api/v1/agent/session/{sess_id}/history")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert hist_data["session_id"] == sess_id
    assert hist_data["history_length"] == 2
    assert len(hist_data["messages"]) == 2
    assert hist_data["messages"][0]["role"] == "user"
    assert hist_data["messages"][1]["role"] == "assistant"

    # Step 3: Delete session
    del_res = client.delete(f"/api/v1/agent/session/{sess_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Step 4: Verify history is now cleared
    hist_after = client.get(f"/api/v1/agent/session/{sess_id}/history")
    assert hist_after.status_code == 200
    assert hist_after.json()["history_length"] == 0



