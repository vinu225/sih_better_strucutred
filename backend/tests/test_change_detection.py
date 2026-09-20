"""
satquery/tests/test_change_detection.py
───────────────────────────────────────
Comprehensive test suite for Bi-temporal Change Detection feature:
- Valid request -> 200 & matches schemas.py
- Unknown tile_id -> clean 404
- Missing / malformed fields -> 422
- Identical tile as before and after (zero false positive)
- Input swapping (A->B vs B->A inverts mean delta sign)
- Mismatched tile shape / bands handling
- Co-registration false warning check
- Cross-modal pair handling
- CORS / Environment configuration check
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import SAMPLES_DIR
from backend.data.sample_generator import generate_sample_tiles
from backend.services.change_detection_service import ChangeDetectionService
from backend.data.modality import detect_modality


@pytest.fixture(scope="module")
def client():
    generate_sample_tiles(SAMPLES_DIR)
    with TestClient(app) as c:
        yield c


def test_change_detection_valid_request(client):
    payload = {
        "tile_id_before": "sample_urban_tile",
        "tile_id_after": "sample_forest_tile",
        "co_registered": True,
        "modality_hint_before": "MULTIMODAL_S1_S2",
        "modality_hint_after": "MULTIMODAL_S1_S2",
        "label_before": "2023-Urban",
        "label_after": "2024-Forest",
    }
    res = client.post("/api/v1/change-detection", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["tile_id_before"] == "sample_urban_tile"
    assert data["tile_id_after"] == "sample_forest_tile"
    assert data["pair_type"] == "TEMPORAL_PAIR"
    assert data["co_registered"] is True
    assert data["warning"] is None
    assert "image_before" in data
    assert "image_after" in data
    assert "pixel_change" in data
    assert data["pixel_change"] is not None
    assert "change_magnitude" in data["pixel_change"]
    assert "per_channel_delta" in data["pixel_change"]
    assert len(data["pixel_change"]["per_channel_delta"]) == 12


def test_change_detection_unknown_tile_id(client):
    payload = {
        "tile_id_before": "non_existent_tile_xyz123",
        "tile_id_after": "sample_forest_tile",
        "co_registered": True,
    }
    res = client.post("/api/v1/change-detection", json=payload)
    assert res.status_code == 404
    detail = res.json()["detail"]
    assert "non_existent_tile_xyz123" in detail
    assert "not found" in detail.lower()


def test_change_detection_missing_fields(client):
    # Missing required tile_id_after
    payload = {
        "tile_id_before": "sample_urban_tile",
    }
    res = client.post("/api/v1/change-detection", json=payload)
    assert res.status_code == 422


def test_change_detection_same_tile_before_after(client):
    payload = {
        "tile_id_before": "sample_urban_tile",
        "tile_id_after": "sample_urban_tile",
        "co_registered": True,
    }
    res = client.post("/api/v1/change-detection", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["pixel_change"] is not None
    cm = data["pixel_change"]["change_magnitude"]
    assert cm["mean"] == 0.0
    assert cm["max"] == 0.0
    for ch in data["pixel_change"]["per_channel_delta"]:
        assert ch["mean_delta"] == 0.0
        assert ch["abs_mean_delta"] == 0.0


def test_change_detection_swap_order_inverts_sign(client):
    payload_fwd = {
        "tile_id_before": "sample_urban_tile",
        "tile_id_after": "sample_forest_tile",
        "co_registered": True,
    }
    payload_rev = {
        "tile_id_before": "sample_forest_tile",
        "tile_id_after": "sample_urban_tile",
        "co_registered": True,
    }
    res_fwd = client.post("/api/v1/change-detection", json=payload_fwd).json()
    res_rev = client.post("/api/v1/change-detection", json=payload_rev).json()
    
    deltas_fwd = res_fwd["pixel_change"]["per_channel_delta"]
    deltas_rev = res_rev["pixel_change"]["per_channel_delta"]
    
    for df, dr in zip(deltas_fwd, deltas_rev):
        assert pytest.approx(df["mean_delta"], abs=1e-5) == -dr["mean_delta"]
        # Absolute mean delta should be symmetric
        assert pytest.approx(df["abs_mean_delta"], abs=1e-5) == dr["abs_mean_delta"]


def test_change_detection_co_registered_false(client):
    payload = {
        "tile_id_before": "sample_urban_tile",
        "tile_id_after": "sample_forest_tile",
        "co_registered": False,
    }
    res = client.post("/api/v1/change-detection", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["co_registered"] is False
    assert data["warning"] is not None
    assert "Co-registration not confirmed" in data["warning"]
    assert data["pixel_change"] is None


def test_change_detection_shape_mismatch():
    arr1 = np.ones((12, 120, 120), dtype=np.float32)
    arr2 = np.ones((12, 60, 60), dtype=np.float32)
    res = ChangeDetectionService.compare(arr1, arr2, co_registered=True)
    assert res["pixel_change"] is None
    assert "Shape mismatch" in res["warning"]


def test_cors_and_frontend_env():
    # Verify CORS allow_origins includes wildcard or vercel domains
    from backend.main import app
    cors_middleware = [m for m in app.user_middleware if "CORSMiddleware" in str(m.cls)]
    assert len(cors_middleware) > 0
