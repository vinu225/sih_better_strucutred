"""
satquery/services/change_detection_service.py
──────────────────────────────────────────────
Image-level and (optionally) pixel-level change detection for temporal pairs.

CO-REGISTRATION CAVEAT:
  Pixel-wise difference maps are geographically meaningful ONLY when both
  images are spatially co-registered to the same coordinate grid (same CRS,
  same resolution, same extent). Without co-registration, pixel differences
  reflect sensor misalignment and projection errors — NOT real geographic
  change. The caller MUST explicitly set `co_registered=True` only when
  this precondition is satisfied. By default this is False.

CROSS-MODAL PAIRS:
  Pixel-wise arithmetic across different modalities (e.g. optical − SAR)
  is not performed. Use this service only for same-modality temporal pairs.
  For cross-modal pairs, call per-image analysis separately on each image.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np

from backend.data.modality import InputModality, ModalitySpec


class ChangeDetectionService:
    """
    Computes change statistics between two co-temporal images.

    Intended for TEMPORAL_PAIR inputs (same modality, different acquisition
    date). Not intended for CROSS_MODAL_PAIR inputs.
    """

    CO_REGISTRATION_WARNING = (
        "⚠️ Co-registration not confirmed. Results show per-image summary "
        "statistics only. Pixel-wise change detection requires both images to "
        "be spatially co-registered (same grid, resolution, and extent). "
        "Without co-registration, pixel differences reflect alignment errors, "
        "not geographic change."
    )

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def _to_numpy(
        data: Union[np.ndarray, "torch.Tensor"],  # noqa: F821
    ) -> np.ndarray:
        try:
            import torch
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
        except ImportError:
            pass
        arr = np.array(data, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        return arr

    @staticmethod
    def _per_image_stats(arr: np.ndarray) -> Dict[str, Any]:
        """Compute basic per-channel statistics for a single image."""
        channels = []
        for i in range(arr.shape[0]):
            band = arr[i].ravel()
            channels.append(
                {
                    "channel": i,
                    "mean": float(np.mean(band)),
                    "std": float(np.std(band)),
                    "min": float(np.min(band)),
                    "max": float(np.max(band)),
                }
            )
        return {"shape": list(arr.shape), "channels": channels}

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    @classmethod
    def compare(
        cls,
        before: Union[np.ndarray, "torch.Tensor"],  # noqa: F821
        after: Union[np.ndarray, "torch.Tensor"],   # noqa: F821
        co_registered: bool = False,
        spec_before: Optional[ModalitySpec] = None,
        spec_after: Optional[ModalitySpec] = None,
        label_before: str = "before",
        label_after: str = "after",
    ) -> Dict[str, Any]:
        """
        Compare two images and return change statistics.

        Parameters
        ----------
        before, after:
            Arrays of shape (C, H, W). Should be the same modality.
        co_registered:
            Set to True ONLY when both images are confirmed to share the same
            spatial grid (CRS, resolution, extent). Default: False.
        spec_before, spec_after:
            Optional ModalitySpecs for each image.
        label_before, label_after:
            Labels shown in the output (e.g. "2022-06", "2023-08").

        Returns
        -------
        dict with keys:
            - pair_type: 'TEMPORAL_PAIR' or 'CROSS_MODAL_PAIR'
            - co_registered: bool
            - warning: str or None
            - image_before: per-image stats
            - image_after: per-image stats
            - pixel_change: dict or None (only when co_registered=True and
              same shape and same modality)
        """
        arr_before = cls._to_numpy(before)
        arr_after = cls._to_numpy(after)

        # Determine pair type
        pair_type = "TEMPORAL_PAIR"
        cross_modal_note = None
        if spec_before is not None and spec_after is not None:
            if spec_before.modality != spec_after.modality:
                pair_type = "CROSS_MODAL_PAIR"
                cross_modal_note = (
                    f"Images have different modalities "
                    f"({spec_before.modality.value} vs {spec_after.modality.value}). "
                    "Pixel-wise arithmetic across modalities is not performed. "
                    "Showing per-image statistics only."
                )

        # Per-image stats (always computed)
        stats_before = cls._per_image_stats(arr_before)
        stats_after = cls._per_image_stats(arr_after)

        # Pixel-wise change: only when co_registered AND same shape AND same modality
        pixel_change: Optional[Dict[str, Any]] = None
        warning: Optional[str] = None

        if cross_modal_note:
            warning = cross_modal_note
        elif not co_registered:
            warning = cls.CO_REGISTRATION_WARNING
        else:
            # co_registered = True, same modality
            if arr_before.shape != arr_after.shape:
                warning = (
                    f"Shape mismatch ({arr_before.shape} vs {arr_after.shape}). "
                    "Cannot compute pixel-wise difference. "
                    "Showing per-image statistics only."
                )
            else:
                diff = arr_after.astype(np.float64) - arr_before.astype(np.float64)
                change_magnitude = np.sqrt(np.sum(diff ** 2, axis=0))  # L2 across channels

                per_channel_delta = []
                for i in range(diff.shape[0]):
                    d = diff[i].ravel()
                    per_channel_delta.append(
                        {
                            "channel": i,
                            "mean_delta": float(np.mean(d)),
                            "std_delta": float(np.std(d)),
                            "abs_mean_delta": float(np.mean(np.abs(d))),
                            "max_abs_delta": float(np.max(np.abs(d))),
                        }
                    )

                pixel_change = {
                    "note": (
                        "Pixel-wise difference computed on co-registered images. "
                        "Spatial co-registration was confirmed by the user."
                    ),
                    "change_magnitude": {
                        "mean": float(np.mean(change_magnitude)),
                        "max": float(np.max(change_magnitude)),
                        "std": float(np.std(change_magnitude)),
                    },
                    "per_channel_delta": per_channel_delta,
                    # Raw maps deliberately not included — they can be very large.
                    # The caller may compute them from before/after arrays directly.
                    "raw_maps_note": (
                        "Raw difference maps are not returned here. Compute "
                        "arr_after - arr_before directly for visualisation."
                    ),
                }

        return {
            "pair_type": pair_type,
            "co_registered": co_registered,
            "warning": warning,
            "label_before": label_before,
            "label_after": label_after,
            "image_before": stats_before,
            "image_after": stats_after,
            "pixel_change": pixel_change,
        }
