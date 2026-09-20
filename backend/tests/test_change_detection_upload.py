"""
satquery/tests/test_change_detection_upload.py
───────────────────────────────────────────────
Comprehensive test suite for POST /api/v1/change-detection dual JSON + multipart upload:
- JSON request -> unchanged 200 response
- multipart with two files -> 200, includes tile_id_before/after
- swapped file order inverts sign of mean_delta
- same file for both -> 0 change
- missing second file -> 422
- wrong file type -> 415
- corrupt file -> 400
- oversized file -> 413
- unsupported content type -> 415
- returned tile_ids work when passed to JSON branch
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import SAMPLES_DIR, MAX_UPLOAD_MB
from backend.data.sample_generator import generate_sample_tiles


@pytest.fixture(scope="module")
def client():
    generate_sample_tiles(SAMPLES_DIR)
    with TestClient(app) as c:
        yield c


def _make_test_png(color=(100, 150, 200), size=(64, 64)):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_json_request_unchanged(client):
    payload = {
        "tile_id_before": "sample_urban_tile",
        "tile_id_after": "sample_forest_tile",
        "co_registered": True,
        "label_before": "urban",
        "label_after": "forest",
    }
    res = client.post("/api/v1/change-detection", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["tile_id_before"] == "sample_urban_tile"
    assert data["tile_id_after"] == "sample_forest_tile"
    assert data["pair_type"] == "TEMPORAL_PAIR"
    assert data["co_registered"] is True
    assert data["pixel_change"] is not None


def test_multipart_two_files_success(client):
    img_b = _make_test_png((50, 100, 150))
    img_a = _make_test_png((150, 200, 50))

    files = [
        ("image_before", ("before.png", img_b, "image/png")),
        ("image_after", ("after.png", img_a, "image/png")),
    ]
    data = {
        "co_registered": "true",
        "label_before": "2023-01",
        "label_after": "2024-01",
    }

    res = client.post("/api/v1/change-detection", files=files, data=data)
    assert res.status_code == 200
    resp_data = res.json()
    assert resp_data["tile_id_before"].startswith("upload_")
    assert resp_data["tile_id_after"].startswith("upload_")
    assert resp_data["pair_type"] == "TEMPORAL_PAIR"
    assert resp_data["co_registered"] is True
    assert resp_data["pixel_change"] is not None


def test_multipart_swap_order_inverts_sign(client):
    img_b = _make_test_png((40, 80, 120))
    img_a = _make_test_png((200, 220, 240))

    files_fwd = [
        ("image_before", ("img1.png", img_b, "image/png")),
        ("image_after", ("img2.png", img_a, "image/png")),
    ]
    files_rev = [
        ("image_before", ("img2.png", img_a, "image/png")),
        ("image_after", ("img1.png", img_b, "image/png")),
    ]

    res_fwd = client.post("/api/v1/change-detection", files=files_fwd, data={"co_registered": "true"}).json()
    res_rev = client.post("/api/v1/change-detection", files=files_rev, data={"co_registered": "true"}).json()

    df_fwd = res_fwd["pixel_change"]["per_channel_delta"]
    df_rev = res_rev["pixel_change"]["per_channel_delta"]

    for d1, d2 in zip(df_fwd, df_rev):
        assert pytest.approx(d1["mean_delta"], abs=1e-5) == -d2["mean_delta"]


def test_multipart_same_file_zero_change(client):
    img_bytes = _make_test_png((100, 100, 100))
    files = [
        ("image_before", ("same.png", img_bytes, "image/png")),
        ("image_after", ("same.png", img_bytes, "image/png")),
    ]

    res = client.post("/api/v1/change-detection", files=files, data={"co_registered": "true"})
    assert res.status_code == 200
    data = res.json()
    cm = data["pixel_change"]["change_magnitude"]
    assert pytest.approx(cm["mean"], abs=1e-6) == 0.0
    assert pytest.approx(cm["max"], abs=1e-6) == 0.0


def test_multipart_missing_second_file(client):
    img_b = _make_test_png((50, 50, 50))
    files = [
        ("image_before", ("before.png", img_b, "image/png")),
    ]
    res = client.post("/api/v1/change-detection", files=files)
    assert res.status_code == 422


def test_multipart_wrong_file_type(client):
    text_content = b"This is a text file, not an image."
    img_b = _make_test_png((50, 50, 50))
    files = [
        ("image_before", ("before.png", img_b, "image/png")),
        ("image_after", ("after.txt", text_content, "text/plain")),
    ]
    res = client.post("/api/v1/change-detection", files=files)
    assert res.status_code == 415
    assert "Unsupported format" in res.json()["detail"]


def test_multipart_corrupt_file(client):
    # Extension .png but header is garbage bytes (magic bytes check fails or decoder fails)
    fake_png = b"\x89PNG\r\n\x1a\nCorruptImageContentGarbageBytesHere"
    img_b = _make_test_png((50, 50, 50))
    files = [
        ("image_before", ("before.png", img_b, "image/png")),
        ("image_after", ("corrupt.png", fake_png, "image/png")),
    ]
    res = client.post("/api/v1/change-detection", files=files)
    assert res.status_code == 400
    assert "Corrupt" in res.json()["detail"] or "unreadable" in res.json()["detail"]


def test_multipart_oversized_file(client, monkeypatch):
    # Temporarily set max size limit to 1MB for test
    import backend.routes as routes_mod
    monkeypatch.setattr(routes_mod, "MAX_UPLOAD_MB", 1)

    big_content = b"\x89PNG\r\n\x1a\n" + b"0" * (2 * 1024 * 1024)
    img_b = _make_test_png((50, 50, 50))
    files = [
        ("image_before", ("before.png", img_b, "image/png")),
        ("image_after", ("big.png", big_content, "image/png")),
    ]
    res = client.post("/api/v1/change-detection", files=files)
    assert res.status_code == 413
    assert "exceeds maximum limit" in res.json()["detail"]


def test_unsupported_content_type(client):
    res = client.post(
        "/api/v1/change-detection",
        content="some plain text content",
        headers={"content-type": "text/plain"},
    )
    assert res.status_code == 415
    assert "Unsupported Content-Type" in res.json()["detail"]


def test_returned_tile_ids_work_in_json_branch(client):
    img_b = _make_test_png((80, 120, 160))
    img_a = _make_test_png((160, 200, 240))

    files = [
        ("image_before", ("img1.png", img_b, "image/png")),
        ("image_after", ("img2.png", img_a, "image/png")),
    ]
    res_mp = client.post("/api/v1/change-detection", files=files, data={"co_registered": "true"})
    assert res_mp.status_code == 200
    mp_data = res_mp.json()

    tid_b = mp_data["tile_id_before"]
    tid_a = mp_data["tile_id_after"]

    # Now pass these returned tile_ids into the JSON branch
    payload = {
        "tile_id_before": tid_b,
        "tile_id_after": tid_a,
        "co_registered": True,
    }
    res_json = client.post("/api/v1/change-detection", json=payload)
    assert res_json.status_code == 200
    json_data = res_json.json()

    assert json_data["tile_id_before"] == tid_b
    assert json_data["tile_id_after"] == tid_a
    assert pytest.approx(json_data["pixel_change"]["change_magnitude"]["mean"]) == mp_data["pixel_change"]["change_magnitude"]["mean"]


def test_openapi_contains_multipart_schema(client):
    res = client.get("/openapi.json")
    assert res.status_code == 200
    schema = res.json()
    cd_op = schema["paths"]["/api/v1/change-detection"]["post"]
    content = cd_op["requestBody"]["content"]

    assert "application/json" in content
    assert "multipart/form-data" in content

    mp_props = content["multipart/form-data"]["schema"]["properties"]
    assert "image_before" in mp_props
    assert "image_after" in mp_props
    assert mp_props["image_before"]["format"] == "binary"
    assert mp_props["image_after"]["format"] == "binary"


def test_dedicated_upload_route_success(client):
    img_b = _make_test_png((60, 120, 180))
    img_a = _make_test_png((180, 120, 60))

    files = [
        ("image_before", ("before_tile.png", img_b, "image/png")),
        ("image_after", ("after_tile.png", img_a, "image/png")),
    ]
    data = {
        "co_registered": "true",
        "label_before": "2023-baseline",
        "label_after": "2024-observed",
    }

    res = client.post("/api/v1/change-detection/upload", files=files, data=data)
    assert res.status_code == 200
    resp_data = res.json()
    assert resp_data["tile_id_before"].startswith("upload_")
    assert resp_data["tile_id_after"].startswith("upload_")
    assert resp_data["pair_type"] == "TEMPORAL_PAIR"
    assert resp_data["co_registered"] is True
    assert resp_data["pixel_change"] is not None


