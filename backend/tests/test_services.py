import torch
import numpy as np
from backend.services.spectral_service import SpectralService
from backend.services.classification_service import ClassificationService
from backend.services.vlm_service import VLMService
from backend.data.dummy_dataset import create_synthetic_signature


def test_spectral_service_indices():
    # Test on forest signature (should have positive NDVI)
    forest_tensor = create_synthetic_signature("forest", 64, 64)
    analysis = SpectralService.analyze_tile(forest_tensor)

    assert "ndvi_stats" in analysis
    assert "ndwi_stats" in analysis
    assert "ndbi_stats" in analysis
    assert "coverage_estimates" in analysis

    assert analysis["ndvi_stats"]["mean"] > 0.3
    assert analysis["coverage_estimates"]["dense_vegetation_percent"] > 50.0


def test_classification_service():
    water_tensor = create_synthetic_signature("water", 64, 64)
    res = ClassificationService.classify_tile(water_tensor, top_k=5)

    assert "primary_class" in res
    assert "confidence" in res or "primary_confidence" in res
    assert len(res["top_predictions"]) == 5
    assert any("water" in p["class_name"].lower() for p in res["top_predictions"])


def test_vlm_service():
    tensor = create_synthetic_signature("cropland", 64, 64)
    res = VLMService.query(tensor, "What crop or land pattern is visible?")

    assert "question" in res
    assert "answer" in res
    assert "primary_landcover" in res
    assert len(res["answer"]) > 0
