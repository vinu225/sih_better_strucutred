"""
satquery/services/sar_service.py
─────────────────────────────────
Basic SAR/backscatter statistics service.

SCOPE LIMITATION:
  This service provides ONLY elementary backscatter statistics derived from
  the raw intensity values of a SAR array. It does NOT perform scientifically
  validated surface property estimation (e.g. surface roughness, soil moisture
  inversion). Any such characterisation would require calibrated SAR data and
  validated retrieval algorithms that are not implemented here.

  Results are labelled accordingly in the output dict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from backend.data.modality import (
    InputModality,
    ModalityError,
    ModalitySpec,
    require_analysis,
)


class SARService:
    """
    Computes basic backscatter statistics from a SAR array.

    Supported input shapes:
        (1, H, W)  — single-polarisation (VH or VV)
        (2, H, W)  — dual-polarisation (VH + VV, e.g. channels 10 & 11 from
                      the 12-ch S1+S2 array)

    This service does NOT call the 12-channel SatelliteEncoder.
    It does NOT pad a SAR-only array to 12 channels.
    """

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

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    @classmethod
    def analyze(
        cls,
        data: Union[np.ndarray, "torch.Tensor"],  # noqa: F821
        spec: Optional[ModalitySpec] = None,
        channel_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute basic backscatter statistics for a SAR array.

        Parameters
        ----------
        data:
            Array of shape (C, H, W) where C is 1 or 2.
            For the 12-ch multimodal case, pass only the SAR channels
            (arr[10:12]) — do NOT pass the full 12-channel array.
        spec:
            Optional ModalitySpec; used only for modality guard when provided.
        channel_labels:
            Optional list of band names per channel (e.g. ["VH", "VV"]).
            Falls back to ["ch_0", "ch_1", ...] if not provided.

        Returns
        -------
        dict with keys:
            - scope_note: str  (always present — explains analysis limitations)
            - channels: list of per-channel dicts with mean/std/min/max
            - polarization_ratio: dict or None  (only for dual-pol inputs)
        """
        # Guard: only call this for SAR-compatible modalities
        if spec is not None:
            allowed = {InputModality.SAR_ONLY, InputModality.MULTIMODAL_S1_S2}
            if spec.modality not in allowed:
                raise ModalityError(
                    "sar_backscatter",
                    spec.modality,
                    "SAR analysis requires SAR_ONLY or MULTIMODAL_S1_S2 input.",
                )

        arr = cls._to_numpy(data)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]  # treat as single-channel

        num_channels = arr.shape[0]

        # Channel labels
        if channel_labels is None:
            channel_labels = [f"ch_{i}" for i in range(num_channels)]
        channel_labels = list(channel_labels)[:num_channels]
        while len(channel_labels) < num_channels:
            channel_labels.append(f"ch_{len(channel_labels)}")

        # Per-channel statistics
        channels: List[Dict[str, Any]] = []
        for i in range(num_channels):
            band = arr[i].astype(np.float32).ravel()
            hist_counts, hist_edges = np.histogram(band, bins=32)
            channels.append(
                {
                    "label": channel_labels[i],
                    "mean": float(np.mean(band)),
                    "std": float(np.std(band)),
                    "min": float(np.min(band)),
                    "max": float(np.max(band)),
                    "median": float(np.median(band)),
                    "histogram": {
                        "counts": hist_counts.tolist(),
                        "bin_edges": hist_edges.tolist(),
                    },
                }
            )

        # Dual-polarisation ratio (VH / VV) — only meaningful when both present
        pol_ratio: Optional[Dict[str, Any]] = None
        if num_channels >= 2:
            vh = arr[0].astype(np.float32)
            vv = arr[1].astype(np.float32)
            denom = np.abs(vv)
            denom[denom < 1e-9] = 1e-9  # avoid division by zero
            ratio = vh / denom
            pol_ratio = {
                "label": f"{channel_labels[0]} / {channel_labels[1]}",
                "mean_ratio": float(np.mean(ratio)),
                "std_ratio": float(np.std(ratio)),
                "note": (
                    "Polarisation ratio computed from raw intensity values. "
                    "Meaningful only for calibrated, co-polarised SAR data."
                ),
            }

        return {
            "scope_note": (
                "Basic backscatter statistics only. "
                "This does NOT constitute scientifically validated surface "
                "property estimation (e.g. roughness, soil moisture). "
                "Interpretation requires domain expertise and calibrated data."
            ),
            "num_channels": num_channels,
            "channel_labels": channel_labels,
            "channels": channels,
            "polarization_ratio": pol_ratio,
        }

    @classmethod
    def extract_sar_channels(cls, arr_12ch: np.ndarray) -> np.ndarray:
        """
        Extract the 2 SAR channels from a 12-channel S1+S2 array (v0.1.1 order).
        Channels 10 (VH) and 11 (VV) in the v0.1.1 spec.

        Parameters
        ----------
        arr_12ch:
            Array of shape (12, H, W).

        Returns
        -------
        Array of shape (2, H, W) containing [VH, VV].
        """
        if arr_12ch.shape[0] != 12:
            raise ValueError(
                f"Expected 12-channel array; got shape {arr_12ch.shape}. "
                "Use this method only with MULTIMODAL_S1_S2 data."
            )
        return arr_12ch[10:12]  # VH (10), VV (11)
