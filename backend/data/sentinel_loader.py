import io
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.config import IMAGE_SIZE, NUM_CHANNELS, BAND_NAMES


def resize_tensor(tensor: torch.Tensor, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Resize [C, H, W] or [B, C, H, W] tensor using native PyTorch interpolation."""
    if tensor.dim() == 3:
        return F.interpolate(tensor.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
    elif tensor.dim() == 4:
        return F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor


def stretch_composite(composite: np.ndarray, lower_pct: float = 2.0, upper_pct: float = 98.0) -> np.ndarray:
    """
    Apply standard remote sensing 2%-98% percentile linear contrast stretch.
    """
    composite = composite.astype(np.float32)
    stretched = np.zeros_like(composite)
    for c in range(3):
        band = composite[:, :, c]
        valid = band[np.isfinite(band)]
        if len(valid) == 0:
            continue
        p_low = np.percentile(valid, lower_pct)
        p_high = np.percentile(valid, upper_pct)
        if p_high > p_low:
            band_stretched = (band - p_low) / (p_high - p_low) * 255.0
            stretched[:, :, c] = np.clip(band_stretched, 0, 255)
        else:
            stretched[:, :, c] = np.clip(band, 0, 255)
    return stretched.astype(np.uint8)


def render_rgb_composite(
    tensor_12ch: Union[torch.Tensor, np.ndarray],
    red_idx: int = 2,   # B04 (Red)
    green_idx: int = 1, # B03 (Green)
    blue_idx: int = 0,  # B02 (Blue)
) -> Image.Image:
    """
    Render True-Color (RGB) composite from 12-channel BigEarthNet v0.1.1 data.
    Channel order: B02 (0), B03 (1), B04 (2).
    """
    if isinstance(tensor_12ch, torch.Tensor):
        arr = tensor_12ch.detach().cpu().numpy()
    else:
        arr = np.array(tensor_12ch)

    if arr.ndim == 4:
        arr = arr[0]

    num_ch = arr.shape[0]
    r_i = red_idx if red_idx < num_ch else min(2, num_ch - 1)
    g_i = green_idx if green_idx < num_ch else min(1, num_ch - 1)
    b_i = blue_idx if blue_idx < num_ch else 0

    r = arr[r_i]
    g = arr[g_i]
    b = arr[b_i]

    rgb = np.stack([r, g, b], axis=-1)
    stretched = stretch_composite(rgb)
    return Image.fromarray(stretched)


def render_false_color_cir(
    tensor_12ch: Union[torch.Tensor, np.ndarray],
    nir_idx: int = 3,  # B08 (NIR)
    red_idx: int = 2,  # B04 (Red)
    green_idx: int = 1, # B03 (Green)
) -> Image.Image:
    """
    Render Color Infrared (CIR) False Color Composite.
    Highlights vegetation in bright red. Channels: B08 (3), B04 (2), B03 (1).
    """
    if isinstance(tensor_12ch, torch.Tensor):
        arr = tensor_12ch.detach().cpu().numpy()
    else:
        arr = np.array(tensor_12ch)

    if arr.ndim == 4:
        arr = arr[0]

    num_ch = arr.shape[0]
    nir_i = nir_idx if nir_idx < num_ch else num_ch - 1
    r_i = red_idx if red_idx < num_ch else min(2, num_ch - 1)
    g_i = green_idx if green_idx < num_ch else min(1, num_ch - 1)

    cir = np.stack([arr[nir_i], arr[r_i], arr[g_i]], axis=-1)
    stretched = stretch_composite(cir)
    return Image.fromarray(stretched)


def render_swir_composite(
    tensor_12ch: Union[torch.Tensor, np.ndarray],
    swir2_idx: int = 8, # B12 (SWIR-2)
    nir_idx: int = 9,   # B8A (Narrow NIR)
    red_idx: int = 2,   # B04 (Red)
) -> Image.Image:
    """
    Render Shortwave Infrared (SWIR) composite.
    Channels: B12 (8), B8A (9), B04 (2).
    """
    if isinstance(tensor_12ch, torch.Tensor):
        arr = tensor_12ch.detach().cpu().numpy()
    else:
        arr = np.array(tensor_12ch)

    if arr.ndim == 4:
        arr = arr[0]

    num_ch = arr.shape[0]
    swir_i = swir2_idx if swir2_idx < num_ch else num_ch - 1
    nir_i = nir_idx if nir_idx < num_ch else min(7, num_ch - 1)
    r_i = red_idx if red_idx < num_ch else min(3, num_ch - 1)

    swir = np.stack([arr[swir_i], arr[nir_i], arr[r_i]], axis=-1)
    stretched = stretch_composite(swir)
    return Image.fromarray(stretched)


def normalize_multispectral_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize 12-channel multispectral tensor to zero mean, unit variance or [0, 1] range.
    """
    if tensor.dim() == 3:
        # [C, H, W]
        min_v = tensor.amin(dim=(1, 2), keepdim=True)
        max_v = tensor.amax(dim=(1, 2), keepdim=True)
        diff = max_v - min_v
        diff[diff == 0] = 1.0
        return (tensor - min_v) / diff
    elif tensor.dim() == 4:
        # [B, C, H, W]
        min_v = tensor.amin(dim=(2, 3), keepdim=True)
        max_v = tensor.amax(dim=(2, 3), keepdim=True)
        diff = max_v - min_v
        diff[diff == 0] = 1.0
        return (tensor - min_v) / diff
    return tensor


class SentinelDataset(torch.utils.data.Dataset):
    """
    Sentinel-2 multispectral dataset loader.
    """
    def __init__(self, num_samples: int = 10, size: int = IMAGE_SIZE):
        self.num_samples = num_samples
        self.size = size

        self.queries = [
            "What type of land cover is present in this satellite image?",
            "Does this multi-spectral region contain water bodies?",
            "What is the dominant landscape in this sentinel tile?",
            "Assess the vegetative vigor and moisture in this sector.",
        ]
        self.answers = [
            "Mixed forest with agricultural fields nearby.",
            "Yes, contains distinct inland water bodies and riparian zones.",
            "Urban fabric interlaced with industrial infrastructure.",
            "High near-infrared reflectance indicates healthy dense canopy.",
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        # Generate representative multispectral patch
        raw_image = torch.randn(NUM_CHANNELS, self.size, self.size)
        image_resized = resize_tensor(raw_image, self.size)
        q_idx = idx % len(self.queries)
        return {
            "image": image_resized,
            "question": self.queries[q_idx],
            "answer": self.answers[q_idx],
            "patch_id": f"synthetic_tile_{idx:03d}",
        }
