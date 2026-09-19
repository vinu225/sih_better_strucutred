import torch
import pytest
from backend.models.satellite_encoder import SatelliteEncoder
from backend.models.projector import VisionProjector
from backend.models.vlm import SatQueryVLM


def test_satellite_encoder_forward():
    encoder = SatelliteEncoder(use_mock_fallback=True)
    dummy_input = torch.randn(2, 12, 120, 120)
    features = encoder(dummy_input)

    assert features.shape == (2, 640), f"Expected shape (2, 640), got {features.shape}"


def test_vision_projector():
    projector = VisionProjector(input_dim=640, output_dim=896)
    dummy_feat = torch.randn(2, 640)
    projected = projector(dummy_feat)

    assert projected.shape == (2, 896), f"Expected shape (2, 896), got {projected.shape}"


def test_satquery_vlm_forward():
    # Use load_pretrained_llm=False and use_mock_encoder=True for fast, isolated unit test
    vlm = SatQueryVLM(load_pretrained_llm=False, use_mock_encoder=True, device="cpu")
    dummy_img = torch.randn(1, 12, 120, 120)
    question = "What land cover is visible?"

    outputs, emb_shape = vlm(dummy_img, question)
    assert emb_shape[0] == 1
    assert emb_shape[2] == 896

    ans = vlm.generate_answer(dummy_img, question)
    assert isinstance(ans, str)
    assert len(ans) > 0
