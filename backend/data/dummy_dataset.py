import torch
from torch.utils.data import Dataset
from backend.config import IMAGE_SIZE, NUM_CHANNELS


def create_synthetic_signature(terrain_type: str, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> torch.Tensor:
    """
    Generate synthetic 12-channel BigEarthNet v0.1.1 multimodal tensor.
    Channels: [B02, B03, B04, B08, B05, B06, B07, B11, B12, B8A, VH, VV]
    """
    noise = torch.randn(NUM_CHANNELS, height, width) * 0.02

    if terrain_type == "forest":
        # B02(0.04), B03(0.08), B04(0.04), B08(0.45), B05(0.12), B06(0.28), B07(0.35), B11(0.18), B12(0.09), B8A(0.42), VH(-18dB->0.05), VV(-12dB->0.10)
        base = torch.tensor([0.04, 0.08, 0.04, 0.45, 0.12, 0.28, 0.35, 0.18, 0.09, 0.42, 0.05, 0.10]).view(12, 1, 1)
    elif terrain_type == "water":
        # B02(0.10), B03(0.09), B04(0.04), B08(0.01), B05(0.02), B06(0.01), B07(0.01), B11(0.01), B12(0.005), B8A(0.01), VH(0.01), VV(0.02)
        base = torch.tensor([0.10, 0.09, 0.04, 0.01, 0.02, 0.01, 0.01, 0.01, 0.005, 0.01, 0.01, 0.02]).view(12, 1, 1)
    elif terrain_type == "urban":
        # B02(0.18), B03(0.19), B04(0.22), B08(0.25), B05(0.23), B06(0.24), B07(0.24), B11(0.30), B12(0.28), B8A(0.25), VH(0.18), VV(0.25)
        base = torch.tensor([0.18, 0.19, 0.22, 0.25, 0.23, 0.24, 0.24, 0.30, 0.28, 0.25, 0.18, 0.25]).view(12, 1, 1)
    elif terrain_type == "cropland":
        # B02(0.05), B03(0.09), B04(0.05), B08(0.42), B05(0.15), B06(0.32), B07(0.38), B11(0.20), B12(0.11), B8A(0.40), VH(0.08), VV(0.14)
        base = torch.tensor([0.05, 0.09, 0.05, 0.42, 0.15, 0.32, 0.38, 0.20, 0.11, 0.40, 0.08, 0.14]).view(12, 1, 1)
    else:
        # Default random
        return torch.rand(NUM_CHANNELS, height, width)

    tensor = base + noise
    return torch.clamp(tensor, 0.001, 1.0)


def get_dummy_batch(batch_size: int = 1, channels: int = NUM_CHANNELS, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> torch.Tensor:
    """
    Generate synthetic batch for rapid pipeline sanity checking.
    """
    return torch.randn(batch_size, channels, height, width)


class DummyMultispectralDataset(Dataset):
    """
    Synthetic dataset containing labeled multispectral samples.
    """
    TERRAINS = ["forest", "water", "urban", "cropland"]

    def __init__(self, size: int = IMAGE_SIZE, samples_per_class: int = 5):
        self.size = size
        self.samples = []
        for terrain in self.TERRAINS:
            for i in range(samples_per_class):
                self.samples.append((terrain, f"{terrain}_{i:02d}"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        terrain, name = self.samples[idx]
        image = create_synthetic_signature(terrain, self.size, self.size)
        return {
            "image": image,
            "terrain": terrain,
            "name": name,
            "question": f"Identify the dominant landscape and features in {name}.",
            "ground_truth": f"The region features {terrain} land cover.",
        }
