"""
satquery/tests/test_agentic_router.py
─────────────────────────────────────
Comprehensive test suite for SatQuery AI Agentic Router and Specialists.
Covers:
- Query-driven intent routing across diverse features (water, trees, buildings, roads, agriculture, land cover)
- Modality validation and sensor integrity guards (rejecting NDVI on RGB)
- Separate RGB vision path (never touching 12-channel S1+S2 encoder)
- Grounding & spatial localization with bounding box generation
- Temporal Before/After change detection
- Optical + SAR joint cross-modal reasoning
- Structured execution trace generation
- API /analyze endpoint verification
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.agent.sat_agent import SatQueryAgent
from backend.agent.tools import ToolRegistry
from backend.data.modality import InputModality, detect_modality, standardize_image_input
from backend.main import app
from backend.config import SAMPLES_DIR
from backend.data.sample_generator import generate_sample_tiles


@pytest.fixture(scope="module")
def agent():
    return SatQueryAgent()


@pytest.fixture(scope="module")
def rgb_array():
    """Create a 3-channel RGB array (3, 120, 120)."""
    np.random.seed(42)
    # Background
    arr = np.ones((3, 120, 120), dtype=np.float32) * 0.3
    # Add green forest patch
    arr[1, 10:40, 10:40] = 0.8
    # Add blue water patch
    arr[2, 60:90, 60:90] = 0.9
    arr[0, 60:90, 60:90] = 0.1
    # Add bright building patch
    arr[:, 80:110, 10:30] = 0.75
    return arr


@pytest.fixture(scope="module")
def s1_s2_array():
    """Create a 12-channel BigEarthNet S1+S2 array."""
    np.random.seed(42)
    return np.random.uniform(0.05, 0.4, (12, 120, 120)).astype(np.float32)


@pytest.fixture(scope="module")
def sar_array():
    """Create a 2-channel SAR array (VH, VV)."""
    return np.random.uniform(0.02, 0.3, (2, 120, 120)).astype(np.float32)


@pytest.fixture(scope="module")
def client():
    generate_sample_tiles(SAMPLES_DIR)
    with TestClient(app) as c:
        yield c


# ==============================================================================
# 1. Modality Validation & Tool Registry Tests
# ==============================================================================

def test_tool_registry_metadata():
    reg = ToolRegistry()
    tools = reg.list_tools()
    assert len(tools) >= 7

    spec_tool = reg.get_tool("spectral_analysis")
    assert spec_tool is not None
    assert InputModality.MULTIMODAL_S1_S2 in spec_tool.supported_modalities
    assert InputModality.RGB_OPTICAL not in spec_tool.supported_modalities

    rgb_tool = reg.get_tool("rgb_scene_analysis")
    assert rgb_tool is not None
    assert InputModality.RGB_OPTICAL in rgb_tool.supported_modalities


def test_modality_detection_rgb(rgb_array):
    spec = detect_modality(rgb_array)
    assert spec.modality == InputModality.RGB_OPTICAL
    assert spec.channel_count == 3


def test_modality_detection_s1_s2(s1_s2_array):
    spec = detect_modality(s1_s2_array)
    assert spec.modality == InputModality.MULTIMODAL_S1_S2
    assert spec.channel_count == 12


def test_modality_detection_sar(sar_array):
    spec = detect_modality(sar_array, metadata={"band_names": ["VH", "VV"]})
    assert spec.modality == InputModality.SAR_ONLY


# ==============================================================================
# 2. Sensor Integrity & Physical Capability Guard Tests
# ==============================================================================

def test_reject_ndvi_on_rgb(agent, rgb_array):
    """
    CRITICAL TEST: Ordinary RGB must NOT calculate NDVI or fake 12 channels.
    Must return clear capability limitation.
    """
    res = agent.analyze(rgb_array, query="Calculate NDVI vegetation index for this image.")
    assert res["selected_task"] == "spectral_analysis"
    assert res["selected_model_or_tool"] == "SensorIntegrityGuard"
    assert "Capability Limitation" in res["answer"]
    assert "Near-Infrared" in res["answer"]
    assert res["confidence"] == 1.0
    assert len(res["execution_trace"]) >= 3


def test_reject_sar_on_rgb(agent, rgb_array):
    """Optical image must NOT fake SAR radar backscatter."""
    res = agent.analyze(rgb_array, query="Assess SAR radar backscatter and polarization ratio.")
    assert res["selected_task"] == "sar_analysis"
    assert res["selected_model_or_tool"] == "SensorIntegrityGuard"
    assert "Capability Limitation" in res["answer"]


# ==============================================================================
# 3. Grounding & Spatial Localization Tests (Diverse Concepts)
# ==============================================================================

def test_grounding_water_bodies(agent, rgb_array):
    res = agent.analyze(rgb_array, query="Locate all water bodies in this region.")
    assert res["selected_task"] == "grounding"
    assert res["selected_model_or_tool"] == "grounding_localization"
    assert res["visual_evidence"] is not None
    assert "image_base64" in res["visual_evidence"]
    assert "boxes" in res["visual_evidence"]
    assert len(res["execution_trace"]) > 0


def test_grounding_trees_forest(agent, rgb_array):
    res = agent.analyze(rgb_array, query="Find dense tree canopy and forest patches.")
    assert res["selected_task"] == "grounding"
    assert res["visual_evidence"] is not None
    assert "image_base64" in res["visual_evidence"]


def test_grounding_buildings_urban(agent, rgb_array):
    res = agent.analyze(rgb_array, query="Where are the buildings located? Count them.")
    assert res["selected_task"] == "grounding"
    assert res["visual_evidence"] is not None
    assert "boxes" in res["visual_evidence"]


def test_grounding_roads(agent, rgb_array):
    res = agent.analyze(rgb_array, query="Detect and highlight road corridors.")
    assert res["selected_task"] == "grounding"
    assert res["visual_evidence"] is not None


def test_grounding_agriculture(agent, rgb_array):
    res = agent.analyze(rgb_array, query="Identify agricultural fields and cropland.")
    assert res["selected_task"] == "grounding"
    assert res["visual_evidence"] is not None


# ==============================================================================
# 4. Separate RGB Path vs S1+S2 Real MobileViT-s Tests
# ==============================================================================

def test_rgb_landcover_separate_path(agent, rgb_array):
    """RGB image must use RGB vision classifier and never 12-channel MobileViT."""
    res = agent.analyze(rgb_array, query="What is the land cover classification?")
    assert res["selected_task"] == "classification"
    assert res["selected_model_or_tool"] == "RGBVisionClassifier"
    assert res["confidence"] > 0


def test_s1_s2_real_mobilevit_path(agent, s1_s2_array):
    """S1+S2 12-channel input must route to BigEarthNet MobileViT-s."""
    res = agent.analyze(s1_s2_array, query="Classify the land cover for this tile.")
    assert res["selected_task"] == "classification"
    assert "MobileViT" in res["selected_model_or_tool"]
    assert res["confidence"] > 0


# ==============================================================================
# 5. Temporal Pair & Cross-Modal Pair Tests
# ==============================================================================

def test_temporal_change_detection(agent, rgb_array):
    """Two images of same modality must route to change detection."""
    after_arr = rgb_array.copy()
    after_arr[:, 50:70, 50:70] += 0.3
    res = agent.analyze([rgb_array, after_arr], query="Compare Before and After images.")
    assert res["selected_task"] == "change_detection"
    assert res["selected_model_or_tool"] == "change_detection"
    assert "Temporal change analysis" in res["answer"]
    assert res["detected_modality"] == "TEMPORAL_PAIR"


def test_cross_modal_optical_sar_joint(agent, rgb_array, sar_array):
    """Optical + SAR pair must route to cross-modal joint reasoning."""
    res = agent.analyze([rgb_array, sar_array], query="Analyze joint optical and SAR radar evidence.")
    assert res["selected_task"] == "cross_modal_analysis"
    assert res["selected_model_or_tool"] == "cross_modal_analysis"
    assert "Cross-Modal Optical + SAR Analysis" in res["answer"]
    assert res["detected_modality"] == "CROSS_MODAL_PAIR"
    assert "image_base64" in res["visual_evidence"]


# ==============================================================================
# 6. Structured Schema & API Endpoint Verification
# ==============================================================================

def test_api_analyze_endpoint(client):
    res = client.post(
        "/api/v1/analyze",
        json={
            "tile_id": "sample_forest_tile",
            "query": "Where are the forest and tree canopy areas?",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "confidence" in data
    assert "visual_evidence" in data
    assert "selected_task" in data
    assert "selected_model_or_tool" in data
    assert "detected_modality" in data
    assert "execution_trace" in data
    assert len(data["execution_trace"]) >= 4


# ==============================================================================
# 7. Explicit 8 Target Scenarios for Single Unified /analyze
# ==============================================================================

def test_sih_scenario_1_rgb_water(agent, rgb_array):
    """TEST 1: RGB JPG/PNG + 'Where are the water bodies?'"""
    res = agent.analyze(rgb_array, query="Where are the water bodies?")
    assert res["detected_modality"] == "RGB_OPTICAL"
    assert res["selected_task"] == "grounding"
    assert res["selected_model_or_tool"] == "grounding_localization"
    assert "water" in res["answer"].lower()
    assert res["confidence"] > 0
    assert "boxes" in res["visual_evidence"]
    assert any("water bodies" in t.lower() or "water" in t.lower() for t in res["execution_trace"])


def test_sih_scenario_2_rgb_describe(agent, rgb_array):
    """TEST 2: RGB JPG/PNG + 'Describe the major objects visible in this image.'"""
    res = agent.analyze(rgb_array, query="Describe the major objects visible in this image.")
    assert res["detected_modality"] == "RGB_OPTICAL"
    assert res["selected_task"] == "rgb_scene_analysis"
    assert res["selected_model_or_tool"] == "RGBVisionService"
    assert "coverage" in res["answer"].lower() or "optical" in res["answer"].lower()


def test_sih_scenario_3_sentinel2_landcover(agent, s1_s2_array):
    """TEST 3: Sentinel-2 multispectral + 'What land-cover types are present?'"""
    res = agent.analyze(s1_s2_array, query="What land-cover types are present?")
    assert res["selected_task"] == "classification"
    assert "MobileViT" in res["selected_model_or_tool"] or "Classifier" in res["selected_model_or_tool"]
    assert res["confidence"] > 0


def test_sih_scenario_4_sar_structures(agent, sar_array):
    """TEST 4: SAR input + 'Analyze the major structures in this SAR image.'"""
    res = agent.analyze(sar_array, query="Analyze the major structures in this SAR image.")
    assert res["detected_modality"] == "SAR_ONLY"
    assert res["selected_task"] == "sar_analysis"
    assert res["selected_model_or_tool"] == "SARService"
    assert "Backscatter" in res["answer"] or "SAR" in res["answer"]


def test_sih_scenario_5_optical_sar_pair(agent, rgb_array, sar_array):
    """TEST 5: Optical + SAR pair + 'Use both images to identify built-up and water-covered regions.'"""
    res = agent.analyze([rgb_array, sar_array], query="Use both images to identify built-up and water-covered regions.")
    assert res["detected_modality"] == "CROSS_MODAL_PAIR"
    assert res["selected_task"] == "cross_modal_analysis"
    assert res["selected_model_or_tool"] == "cross_modal_analysis"
    assert "Cross-Modal" in res["answer"]
    assert "image_base64" in res["visual_evidence"]


def test_sih_scenario_6_temporal_pair(agent, rgb_array):
    """TEST 6: Before + After pair + 'What changed between these two images?'"""
    after_arr = rgb_array.copy()
    after_arr[:, 30:60, 30:60] += 0.2
    res = agent.analyze([rgb_array, after_arr], query="What changed between these two images?")
    assert res["detected_modality"] == "TEMPORAL_PAIR"
    assert res["selected_task"] == "change_detection"
    assert res["selected_model_or_tool"] == "change_detection"
    assert "change" in res["answer"].lower()
    assert "image_base64" in res["visual_evidence"]


def test_sih_scenario_7_rgb_ndvi_rejection(agent, rgb_array):
    """TEST 7: RGB + 'Calculate NDVI.' -> safe rejection"""
    res = agent.analyze(rgb_array, query="Calculate NDVI.")
    assert res["selected_task"] == "spectral_analysis"
    assert res["selected_model_or_tool"] == "SensorIntegrityGuard"
    assert "Capability Limitation" in res["answer"]
    assert "Near-Infrared" in res["answer"]


def test_sih_scenario_8_sar_classification_guard(agent, sar_array):
    """TEST 8: Standalone SAR + 'Classify the land cover for this tile.' -> guarded rejection"""
    res = agent.analyze(sar_array, query="Classify the land cover for this tile.")
    assert res["selected_task"] == "classification"
    assert res["selected_model_or_tool"] == "SensorIntegrityGuard"
    assert "Capability Limitation" in res["answer"]
    assert "12-channel" in res["answer"]

