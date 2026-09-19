import torch
import numpy as np
from PIL import Image
from backend.data.sentinel_loader import (
    SentinelDataset,
    render_rgb_composite,
    render_false_color_cir,
    render_swir_composite,
    normalize_multispectral_tensor,
)
from backend.data.dummy_dataset import DummyMultispectralDataset, create_synthetic_signature


def test_sentinel_dataset():
    dataset = SentinelDataset(num_samples=4, size=120)
    assert len(dataset) == 4
    item = dataset[0]
    assert "image" in item
    assert item["image"].shape == (12, 120, 120)
    assert "question" in item
    assert "answer" in item


def test_dummy_multispectral_signatures():
    for terrain in ["forest", "water", "urban", "cropland"]:
        tensor = create_synthetic_signature(terrain, 120, 120)
        assert tensor.shape == (12, 120, 120)
        assert torch.all(tensor >= 0.0)


def test_composite_rendering():
    dummy = torch.randn(12, 64, 64)
    rgb = render_rgb_composite(dummy)
    assert isinstance(rgb, Image.Image)
    assert rgb.size == (64, 64)

    cir = render_false_color_cir(dummy)
    assert isinstance(cir, Image.Image)

    swir = render_swir_composite(dummy)
    assert isinstance(swir, Image.Image)


def test_normalization():
    raw = torch.randn(12, 32, 32) * 50 + 100
    norm = normalize_multispectral_tensor(raw)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
