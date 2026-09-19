"""
satquery/data/modality.py
─────────────────────────
Modality detection and routing helpers for SatQuery.

Detection precedence (highest → lowest):
  1. user_hint  – user explicitly selected a modality via UI
  2. metadata   – band_names / modality key in a sidecar dict
  3. shape      – channel-count heuristic (weak, always marked "unconfirmed")

IMPORTANT: Channel count alone is NOT definitive.
  A 1-channel array may be SAR, NIR, thermal, or grayscale optical.
  A 3-channel array may be RGB or three Sentinel-2 bands.
  Always show the detection_basis to the user and allow override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Modality enum
# ---------------------------------------------------------------------------

class InputModality(str, Enum):
    """Recognised input modalities for SatQuery."""

    MULTIMODAL_S1_S2 = "MULTIMODAL_S1_S2"
    """10 Sentinel-2 optical bands + 2 Sentinel-1 SAR channels (12-ch, v0.1.1)."""

    SENTINEL2_MULTISPECTRAL = "SENTINEL2_MULTISPECTRAL"
    """Sentinel-2 spectral bands only (no SAR, 2–10 channels)."""

    SAR_ONLY = "SAR_ONLY"
    """Sentinel-1 SAR data (VH, VV, or single-polarisation, 1–2 channels)."""

    NIR_INFRARED = "NIR_INFRARED"
    """Near-infrared or thermal single-channel data."""

    RGB_OPTICAL = "RGB_OPTICAL"
    """Standard 3-channel optical or aerial imagery (R, G, B)."""

    TEMPORAL_PAIR = "TEMPORAL_PAIR"
    """Two images of the *same* modality captured at different times (before/after)."""

    CROSS_MODAL_PAIR = "CROSS_MODAL_PAIR"
    """Two images of *different* modalities (e.g. optical + SAR)."""

    UNKNOWN = "UNKNOWN"
    """Unrecognised shape or configuration."""


# ---------------------------------------------------------------------------
# Analyses each modality supports
# ---------------------------------------------------------------------------

_MODALITY_ANALYSES: Dict[InputModality, List[str]] = {
    InputModality.MULTIMODAL_S1_S2: [
        "rgb_composite",
        "false_color_cir",
        "swir_composite",
        "band_inspector",
        "ndvi",
        "ndwi",
        "ndbi",
        "classification",
        "sar_backscatter",
        "vlm_query",
    ],
    InputModality.SENTINEL2_MULTISPECTRAL: [
        "rgb_composite",       # only if ≥ 3 channels
        "band_inspector",
        "ndvi",                # only if NIR + Red present
        "ndwi",                # only if Green + NIR present
        "ndbi",                # only if SWIR + NIR present
        "vlm_query",
    ],
    InputModality.SAR_ONLY: [
        "band_inspector",
        "sar_backscatter",
        "vlm_query",
    ],
    InputModality.NIR_INFRARED: [
        "band_inspector",
        "vlm_query",
    ],
    InputModality.RGB_OPTICAL: [
        "rgb_composite",
        "band_inspector",
        "rgb_scene_analysis",
        "grounding",
        "classification",
        "vlm_query",
    ],
    InputModality.TEMPORAL_PAIR: [
        # per-image analyses are resolved from constituent modalities at runtime
        "temporal_change",
        "vlm_query",
    ],
    InputModality.CROSS_MODAL_PAIR: [
        # per-image analyses resolved from constituent modalities at runtime
        "cross_modal_analysis",
        "vlm_query",
    ],
    InputModality.UNKNOWN: [],
}


# ---------------------------------------------------------------------------
# ModalityError
# ---------------------------------------------------------------------------

class ModalityError(ValueError):
    """Raised when an analysis is invoked on an incompatible modality."""

    def __init__(self, requested_analysis: str, detected_modality: InputModality, reason: str = ""):
        msg = (
            f"Analysis '{requested_analysis}' is not available for modality "
            f"'{detected_modality.value}'."
        )
        if reason:
            msg += f" {reason}"
        super().__init__(msg)
        self.requested_analysis = requested_analysis
        self.detected_modality = detected_modality
        self.reason = reason


# ---------------------------------------------------------------------------
# ModalitySpec dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModalitySpec:
    """Describes the detected or user-specified input modality."""

    modality: InputModality
    channel_count: int
    detection_basis: str
    """One of: 'user', 'metadata', or 'shape_heuristic (unconfirmed)'."""

    band_names: Optional[List[str]] = None
    """Named bands if known from metadata or v0.1.1 spec; None if unknown."""

    available_analyses: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.available_analyses:
            self.available_analyses = _MODALITY_ANALYSES.get(self.modality, [])

    @property
    def display_label(self) -> str:
        """Short human-readable label for the UI modality banner."""
        _labels = {
            InputModality.MULTIMODAL_S1_S2: "🛰️ MULTIMODAL S1+S2",
            InputModality.SENTINEL2_MULTISPECTRAL: "🌿 SENTINEL-2 MULTISPECTRAL",
            InputModality.SAR_ONLY: "🌊 SAR ONLY",
            InputModality.NIR_INFRARED: "🔴 NIR / INFRARED",
            InputModality.RGB_OPTICAL: "📷 RGB OPTICAL",
            InputModality.TEMPORAL_PAIR: "⏱️ TEMPORAL PAIR (Before/After)",
            InputModality.CROSS_MODAL_PAIR: "🔀 CROSS-MODAL PAIR",
            InputModality.UNKNOWN: "❓ UNKNOWN",
        }
        return _labels.get(self.modality, self.modality.value)

    @property
    def basis_label(self) -> str:
        """Human-readable detection basis for the UI."""
        _basis = {
            "user": "User selected",
            "metadata": "Detected from metadata",
            "shape_heuristic (unconfirmed)": "Detected from channel count (unconfirmed — please verify)",
        }
        return _basis.get(self.detection_basis, self.detection_basis)

    def supports(self, analysis: str) -> bool:
        return analysis in self.available_analyses

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "channel_count": self.channel_count,
            "detection_basis": self.detection_basis,
            "basis_label": self.basis_label,
            "display_label": self.display_label,
            "band_names": self.band_names,
            "available_analyses": self.available_analyses,
        }


# ---------------------------------------------------------------------------
# Shape-heuristic helper (WEAK — always marked unconfirmed)
# ---------------------------------------------------------------------------

# v0.1.1 canonical band names for a 12-channel S1+S2 array
_V011_BAND_NAMES = [
    "B02", "B03", "B04", "B08",
    "B05", "B06", "B07", "B11",
    "B12", "B8A", "VH", "VV",
]


def _guess_from_shape(channel_count: int) -> InputModality:
    """
    Return the most *likely* modality for a given channel count.

    This is a heuristic only. A 1-channel array could be SAR, NIR, thermal,
    or grayscale optical. Do NOT treat this as authoritative.
    """
    if channel_count == 12:
        return InputModality.MULTIMODAL_S1_S2
    if channel_count == 3:
        return InputModality.RGB_OPTICAL
    if channel_count == 2:
        return InputModality.SAR_ONLY
    if channel_count == 1:
        # Could be SAR, NIR, or grayscale — mark as unknown so user must clarify
        return InputModality.UNKNOWN
    if 4 <= channel_count <= 10:
        return InputModality.SENTINEL2_MULTISPECTRAL
    return InputModality.UNKNOWN


def _guess_from_band_names(band_names: List[str]) -> InputModality:
    """Infer modality from a list of named bands (case-insensitive)."""
    names_upper = {n.upper() for n in band_names}
    has_sar = bool(names_upper & {"VH", "VV"})
    has_s2 = bool(names_upper & {"B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"})

    if has_sar and has_s2:
        return InputModality.MULTIMODAL_S1_S2
    if has_sar and not has_s2:
        return InputModality.SAR_ONLY
    if has_s2 and not has_sar:
        return InputModality.SENTINEL2_MULTISPECTRAL
    # RGB convention
    if names_upper <= {"R", "G", "B"} or names_upper <= {"RED", "GREEN", "BLUE"}:
        return InputModality.RGB_OPTICAL
    if names_upper & {"NIR", "B8A", "B08"}:
        return InputModality.NIR_INFRARED
    return InputModality.UNKNOWN


# ---------------------------------------------------------------------------
# Public API: detect_modality
# ---------------------------------------------------------------------------

def detect_modality(
    tensor: Union[np.ndarray, "torch.Tensor"],  # noqa: F821
    metadata: Optional[Dict[str, Any]] = None,
    user_hint: Optional[str] = None,
) -> ModalitySpec:
    """
    Detect the input modality in precedence order:
      1. user_hint   → detection_basis = 'user'
      2. metadata    → detection_basis = 'metadata'
      3. shape       → detection_basis = 'shape_heuristic (unconfirmed)'

    Parameters
    ----------
    tensor:
        Array of shape (C, H, W) or (H, W) or (B, C, H, W).
    metadata:
        Optional dict that may contain:
          - 'modality': str matching an InputModality value
          - 'band_names': list[str]
    user_hint:
        Optional InputModality value string supplied by the user.

    Returns
    -------
    ModalitySpec with resolved modality and detection_basis.
    """
    # Resolve channel count
    arr = np.array(tensor) if not isinstance(tensor, np.ndarray) else tensor
    if arr.ndim == 4:
        channel_count = arr.shape[1]
    elif arr.ndim == 3:
        if arr.shape[2] in (1, 3, 4) and arr.shape[0] > 12:
            channel_count = arr.shape[2]
        else:
            channel_count = arr.shape[0]
    elif arr.ndim == 2:
        channel_count = 1
    else:
        channel_count = 0

    # 1. User override
    if user_hint is not None:
        try:
            modality = InputModality(user_hint)
        except ValueError:
            modality = InputModality.UNKNOWN
        band_names = (
            _V011_BAND_NAMES if modality == InputModality.MULTIMODAL_S1_S2 else None
        )
        return ModalitySpec(
            modality=modality,
            channel_count=channel_count,
            detection_basis="user",
            band_names=band_names,
        )

    # 2. Metadata
    if metadata:
        # Explicit modality key
        if "modality" in metadata:
            try:
                modality = InputModality(metadata["modality"])
            except ValueError:
                modality = InputModality.UNKNOWN
            band_names = metadata.get("band_names")
            return ModalitySpec(
                modality=modality,
                channel_count=channel_count,
                detection_basis="metadata",
                band_names=band_names,
            )

        # Band names key
        if "band_names" in metadata:
            band_names: List[str] = metadata["band_names"]
            modality = _guess_from_band_names(band_names)
            return ModalitySpec(
                modality=modality,
                channel_count=channel_count,
                detection_basis="metadata",
                band_names=band_names,
            )

    # 3. Shape heuristic (always "unconfirmed")
    modality = _guess_from_shape(channel_count)
    band_names = _V011_BAND_NAMES if modality == InputModality.MULTIMODAL_S1_S2 else None
    return ModalitySpec(
        modality=modality,
        channel_count=channel_count,
        detection_basis="shape_heuristic (unconfirmed)",
        band_names=band_names,
    )


# ---------------------------------------------------------------------------
# Pair-level modality resolver
# ---------------------------------------------------------------------------

def resolve_pair_modality(spec_a: ModalitySpec, spec_b: ModalitySpec) -> InputModality:
    """
    Given two individual ModalitySpecs, determine whether the pair is a
    TEMPORAL_PAIR (same sensor/modality, different time) or
    CROSS_MODAL_PAIR (different modalities).

    Note: This uses modality enum equality only.
    """
    if spec_a.modality == spec_b.modality:
        return InputModality.TEMPORAL_PAIR
    return InputModality.CROSS_MODAL_PAIR


# ---------------------------------------------------------------------------
# Guard helper for services
# ---------------------------------------------------------------------------

def require_analysis(spec: ModalitySpec, analysis: str, reason: str = "") -> None:
    """
    Assert that `analysis` is available for the given ModalitySpec.
    Raises ModalityError if not.
    """
    if not spec.supports(analysis):
        raise ModalityError(analysis, spec.modality, reason)


def get_available_analyses(spec: ModalitySpec) -> List[str]:
    """Return list of analysis names available for the given spec."""
    return spec.available_analyses


def standardize_image_input(image_input: Any) -> np.ndarray:
    """
    Standardize diverse image inputs (PIL Image, file path, bytes, torch.Tensor, or ndarray)
    into a channel-first float32 numpy array of shape (C, H, W).
    Normalizes integer inputs to [0.0, 1.0].
    """
    import io
    from pathlib import Path
    from PIL import Image

    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if p.suffix.lower() in [".npy", ".npz"]:
            arr = np.load(str(p))
        else:
            with Image.open(str(p)) as im:
                arr = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
                return np.transpose(arr, (2, 0, 1))

    elif isinstance(image_input, (bytes, bytearray)):
        try:
            buf = io.BytesIO(image_input)
            arr = np.load(buf)
        except Exception:
            buf = io.BytesIO(image_input)
            with Image.open(buf) as im:
                arr = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
                return np.transpose(arr, (2, 0, 1))

    elif isinstance(image_input, Image.Image):
        arr = np.array(image_input.convert("RGB"), dtype=np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))

    elif hasattr(image_input, "detach"):  # torch.Tensor
        arr = image_input.detach().cpu().numpy()
    else:
        arr = np.array(image_input)

    # Dimensionality & channel order normalization
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4) and arr.shape[0] > 12:
        arr = np.transpose(arr, (2, 0, 1))

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)

    # Value range normalization for uint8 images
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)

    return arr


def to_rgb_image(arr: np.ndarray) -> Any:
    """
    Render any satellite / optical / SAR array as a PIL RGB image for overlays or viewing.
    """
    from PIL import Image
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.expand_dims(arr, 0)

    c, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
    if c >= 12:
        # Standard BigEarthNet v0.1.1 optical bands: B04 (Red, ch 2), B03 (Green, ch 1), B02 (Blue, ch 0)
        r, g, b = arr[2], arr[1], arr[0]
        rgb = np.stack([r, g, b], axis=-1)
    elif c >= 3:
        # Standard RGB channels: 0, 1, 2
        rgb = np.stack([arr[0], arr[1], arr[2]], axis=-1)
    elif c == 2:
        # Dual-polarization SAR (VH, VV, VH/VV)
        vh = arr[0]
        vv = arr[1]
        denom = vv.copy()
        denom[denom == 0] = 1e-6
        ratio = np.clip(vh / denom, 0.0, 1.0)
        rgb = np.stack([vh, vv, ratio], axis=-1)
    else:
        # Single channel (grayscale duplicated)
        single = arr[0]
        rgb = np.stack([single, single, single], axis=-1)

    # Robust min-max stretch to 0..255
    p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
    if p98 > p2:
        norm = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
    else:
        norm = np.clip(rgb, 0.0, 1.0) if np.max(rgb) <= 1.0 else np.clip(rgb / 255.0, 0.0, 1.0)

    u8 = (norm * 255.0).astype(np.uint8)
    return Image.fromarray(u8)

