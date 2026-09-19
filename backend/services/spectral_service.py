"""
satquery/services/spectral_service.py
──────────────────────────────────────
Computes Earth Observation spectral indices from multispectral data.

Modality guard:
  Each index method checks whether the required bands are available for the
  given ModalitySpec before computing. If bands are absent, the method raises
  ModalityError. The analyze_tile() method runs all three indices and returns
  a partial result dict where unavailable indices are marked with:
    {"status": "UNAVAILABLE", "reason": "<why>"}
  rather than fabricating values.
"""

from typing import Any, Dict, Optional, Union

import numpy as np

from backend.config import BAND_INDICES
from backend.data.modality import InputModality, ModalityError, ModalitySpec


# Indices that are computable per modality.
# Key: (index_name, required_named_bands_in_spec)
# A band is considered present when either:
#   (a) spec.band_names is None (unknown / user trusts the data), or
#   (b) the required named bands appear in spec.band_names.
_INDEX_REQUIRED_BANDS: Dict[str, list] = {
    "ndvi": ["B08", "B04"],   # NIR, Red
    "ndwi": ["B03", "B08"],   # Green, NIR
    "ndbi": ["B11", "B08"],   # SWIR1, NIR
}

# Modalities for which spectral indices are never meaningful
_SPECTRAL_BLOCKED_MODALITIES = {
    InputModality.RGB_OPTICAL,     # RGB lacks NIR (B08) and SWIR (B11) necessary for valid NDVI/NDWI/NDBI
    InputModality.SAR_ONLY,
    InputModality.UNKNOWN,
    InputModality.CROSS_MODAL_PAIR,
    InputModality.TEMPORAL_PAIR,   # individual images must be analysed separately
}


def _bands_available(index_name: str, spec: Optional[ModalitySpec]) -> tuple[bool, str]:
    """
    Return (available: bool, reason: str).
    If spec is None or spec.band_names is None, we assume bands are present
    (the caller is responsible for providing the correct array).
    """
    if spec is None or spec.band_names is None:
        return True, ""

    required = _INDEX_REQUIRED_BANDS.get(index_name, [])
    missing = [b for b in required if b not in spec.band_names]
    if missing:
        return False, f"Requires bands {required}; missing: {missing}."
    return True, ""


class SpectralService:
    """
    Computes Earth Observation spectral indices and environmental metrics
    from Sentinel-2 or multimodal (S1+S2) imagery.

    Use analyze_tile() for a comprehensive result dict that handles partial
    availability gracefully rather than fabricating values.
    """

    @staticmethod
    def _to_numpy(tensor: Union["torch.Tensor", np.ndarray]) -> np.ndarray:  # noqa: F821
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
        except ImportError:
            pass
        arr = np.array(tensor, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        return arr

    # ------------------------------------------------------------------
    # Modality guard
    # ------------------------------------------------------------------

    @staticmethod
    def _check_modality(spec: Optional[ModalitySpec], analysis: str = "spectral_indices") -> None:
        """
        Raise ModalityError for modalities that cannot support spectral indices
        at all (e.g. SAR-only, unknown).
        """
        if spec is None:
            return
        if spec.modality in _SPECTRAL_BLOCKED_MODALITIES:
            raise ModalityError(
                analysis,
                spec.modality,
                "Spectral indices require Sentinel-2 optical bands which are "
                "not present in this modality.",
            )

    # ------------------------------------------------------------------
    # Individual index methods (raise ModalityError if blocked)
    # ------------------------------------------------------------------

    @classmethod
    def calculate_ndvi(
        cls,
        data: Union["torch.Tensor", np.ndarray],  # noqa: F821
        nir_idx: int = BAND_INDICES.get("B08", 3),
        red_idx: int = BAND_INDICES.get("B04", 2),
        spec: Optional[ModalitySpec] = None,
    ) -> np.ndarray:
        """
        Normalized Difference Vegetation Index (NDVI).
        NDVI = (NIR - Red) / (NIR + Red)

        Raises ModalityError if spec indicates a modality that blocks spectral
        analysis, or if named bands confirm NIR/Red are absent.
        """
        cls._check_modality(spec, "ndvi")
        avail, reason = _bands_available("ndvi", spec)
        if not avail:
            raise ModalityError("ndvi", spec.modality if spec else InputModality.UNKNOWN, reason)

        arr = cls._to_numpy(data)
        nir = arr[nir_idx].astype(np.float32)
        red = arr[red_idx].astype(np.float32)
        denom = nir + red
        denom[denom == 0] = 1e-6
        return np.clip((nir - red) / denom, -1.0, 1.0)

    @classmethod
    def calculate_ndwi(
        cls,
        data: Union["torch.Tensor", np.ndarray],  # noqa: F821
        green_idx: int = BAND_INDICES.get("B03", 1),
        nir_idx: int = BAND_INDICES.get("B08", 3),
        spec: Optional[ModalitySpec] = None,
    ) -> np.ndarray:
        """
        Normalized Difference Water Index (NDWI — McFeeters).
        NDWI = (Green - NIR) / (Green + NIR)
        """
        cls._check_modality(spec, "ndwi")
        avail, reason = _bands_available("ndwi", spec)
        if not avail:
            raise ModalityError("ndwi", spec.modality if spec else InputModality.UNKNOWN, reason)

        arr = cls._to_numpy(data)
        green = arr[green_idx].astype(np.float32)
        nir = arr[nir_idx].astype(np.float32)
        denom = green + nir
        denom[denom == 0] = 1e-6
        return np.clip((green - nir) / denom, -1.0, 1.0)

    @classmethod
    def calculate_ndbi(
        cls,
        data: Union["torch.Tensor", np.ndarray],  # noqa: F821
        swir_idx: int = BAND_INDICES.get("B11", 7),
        nir_idx: int = BAND_INDICES.get("B08", 3),
        spec: Optional[ModalitySpec] = None,
    ) -> np.ndarray:
        """
        Normalized Difference Built-up Index (NDBI).
        NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
        """
        cls._check_modality(spec, "ndbi")
        avail, reason = _bands_available("ndbi", spec)
        if not avail:
            raise ModalityError("ndbi", spec.modality if spec else InputModality.UNKNOWN, reason)

        arr = cls._to_numpy(data)
        swir = arr[swir_idx].astype(np.float32)
        nir = arr[nir_idx].astype(np.float32)
        denom = swir + nir
        denom[denom == 0] = 1e-6
        return np.clip((swir - nir) / denom, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Comprehensive tile analysis — partial-result aware
    # ------------------------------------------------------------------

    @classmethod
    def analyze_tile(
        cls,
        data: Union["torch.Tensor", np.ndarray],  # noqa: F821
        spec: Optional[ModalitySpec] = None,
    ) -> Dict[str, Any]:
        """
        Compute all available spectral metrics.

        For each index, if the required bands are confirmed absent (from spec),
        the result entry is:
            {"status": "UNAVAILABLE", "reason": "..."}
        rather than a fabricated value.

        If spec is None, all indices are attempted (caller asserts correct data).
        """
        # Hard block for incompatible modalities
        if spec is not None and spec.modality in _SPECTRAL_BLOCKED_MODALITIES:
            return {
                "modality_error": True,
                "modality": spec.modality.value,
                "message": (
                    f"Spectral indices are not available for modality "
                    f"'{spec.modality.value}'. "
                    "Sentinel-2 optical bands are required."
                ),
                "ndvi_stats": {"status": "UNAVAILABLE", "reason": f"Modality is {spec.modality.value}"},
                "ndwi_stats": {"status": "UNAVAILABLE", "reason": f"Modality is {spec.modality.value}"},
                "ndbi_stats": {"status": "UNAVAILABLE", "reason": f"Modality is {spec.modality.value}"},
                "coverage_estimates": {"status": "UNAVAILABLE"},
                "assessment": "Spectral analysis unavailable for this modality.",
            }

        result: Dict[str, Any] = {}
        raw_maps: Dict[str, Any] = {}

        # NDVI
        avail_ndvi, reason_ndvi = _bands_available("ndvi", spec)
        if avail_ndvi:
            try:
                ndvi = cls.calculate_ndvi(data, spec=spec)
                result["ndvi_stats"] = {
                    "mean": float(np.mean(ndvi)),
                    "min": float(np.min(ndvi)),
                    "max": float(np.max(ndvi)),
                    "std": float(np.std(ndvi)),
                }
                raw_maps["ndvi"] = ndvi
            except (IndexError, ModalityError) as e:
                result["ndvi_stats"] = {"status": "UNAVAILABLE", "reason": str(e)}
        else:
            result["ndvi_stats"] = {"status": "UNAVAILABLE", "reason": reason_ndvi}

        # NDWI
        avail_ndwi, reason_ndwi = _bands_available("ndwi", spec)
        if avail_ndwi:
            try:
                ndwi = cls.calculate_ndwi(data, spec=spec)
                result["ndwi_stats"] = {
                    "mean": float(np.mean(ndwi)),
                    "min": float(np.min(ndwi)),
                    "max": float(np.max(ndwi)),
                    "std": float(np.std(ndwi)),
                }
                raw_maps["ndwi"] = ndwi
            except (IndexError, ModalityError) as e:
                result["ndwi_stats"] = {"status": "UNAVAILABLE", "reason": str(e)}
        else:
            result["ndwi_stats"] = {"status": "UNAVAILABLE", "reason": reason_ndwi}

        # NDBI
        avail_ndbi, reason_ndbi = _bands_available("ndbi", spec)
        if avail_ndbi:
            try:
                ndbi = cls.calculate_ndbi(data, spec=spec)
                result["ndbi_stats"] = {
                    "mean": float(np.mean(ndbi)),
                    "min": float(np.min(ndbi)),
                    "max": float(np.max(ndbi)),
                    "std": float(np.std(ndbi)),
                }
                raw_maps["ndbi"] = ndbi
            except (IndexError, ModalityError) as e:
                result["ndbi_stats"] = {"status": "UNAVAILABLE", "reason": str(e)}
        else:
            result["ndbi_stats"] = {"status": "UNAVAILABLE", "reason": reason_ndbi}

        # Coverage estimates (only when NDVI and NDWI available)
        ndvi_arr = raw_maps.get("ndvi")
        ndwi_arr = raw_maps.get("ndwi")
        ndbi_arr = raw_maps.get("ndbi")

        if ndvi_arr is not None:
            total_pixels = ndvi_arr.size
            dense_veg_pct = float(np.sum(ndvi_arr > 0.4) / total_pixels * 100.0)
            water_pct = float(np.sum(ndwi_arr > 0.0) / total_pixels * 100.0) if ndwi_arr is not None else None
            builtup_pct = float(np.sum(ndbi_arr > 0.0) / total_pixels * 100.0) if ndbi_arr is not None else None

            coverage: Dict[str, Any] = {
                "dense_vegetation_percent": round(dense_veg_pct, 2),
            }
            if water_pct is not None:
                coverage["water_body_percent"] = round(water_pct, 2)
            if builtup_pct is not None:
                coverage["builtup_percent"] = round(builtup_pct, 2)

            result["coverage_estimates"] = coverage

            summary_text = []
            if dense_veg_pct > 40:
                summary_text.append(f"Significant vegetative canopy ({dense_veg_pct:.1f}% dense vegetation)")
            if water_pct is not None and water_pct > 15:
                summary_text.append(f"Prominent hydrological surface ({water_pct:.1f}% water)")
            if builtup_pct is not None and builtup_pct > 30:
                summary_text.append(f"Urbanized or impervious terrain ({builtup_pct:.1f}% built-up)")
            if not summary_text:
                summary_text.append("Mixed heterogeneous terrain with moderate vegetative activity")
            result["assessment"] = ". ".join(summary_text) + "."
        else:
            result["coverage_estimates"] = {
                "status": "UNAVAILABLE",
                "reason": "NDVI could not be computed.",
            }
            result["assessment"] = "Spectral assessment unavailable — required bands absent."

        result["raw_maps"] = raw_maps
        return result
