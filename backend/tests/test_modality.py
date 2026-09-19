"""
satquery/tests/test_modality.py
────────────────────────────────
[SYNTHETIC] — Tests routing logic and modality guards only.
No real trained model is loaded. No real satellite data is used.
These tests do NOT verify VQA or model understanding.
"""

import numpy as np
import pytest

from backend.data.modality import (
    InputModality,
    ModalityError,
    ModalitySpec,
    detect_modality,
    require_analysis,
    resolve_pair_modality,
)
from backend.services.spectral_service import SpectralService
from backend.services.sar_service import SARService
from backend.services.change_detection_service import ChangeDetectionService


# ---------------------------------------------------------------------------
# Fixtures — all synthetic tensors
# ---------------------------------------------------------------------------

def make_arr(channels: int, h: int = 16, w: int = 16) -> np.ndarray:
    """Return a zero-filled float32 array of shape (channels, h, w)."""
    return np.zeros((channels, h, w), dtype=np.float32)


# ---------------------------------------------------------------------------
# detect_modality — precedence tests
# ---------------------------------------------------------------------------

class TestDetectModality:
    """[SYNTHETIC] Tests for detect_modality() precedence and heuristics."""

    def test_user_hint_overrides_shape(self):
        """User hint wins regardless of channel count."""
        arr = make_arr(3)
        spec = detect_modality(arr, user_hint="SAR_ONLY")
        assert spec.modality == InputModality.SAR_ONLY
        assert spec.detection_basis == "user"

    def test_metadata_band_names_beat_shape(self):
        """band_names in metadata overrides shape heuristic."""
        arr = make_arr(2)
        spec = detect_modality(arr, metadata={"band_names": ["VH", "VV"]})
        assert spec.modality == InputModality.SAR_ONLY
        assert spec.detection_basis == "metadata"
        assert spec.band_names == ["VH", "VV"]

    def test_metadata_modality_key_beats_shape(self):
        """Explicit 'modality' key in metadata overrides shape heuristic."""
        arr = make_arr(3)
        spec = detect_modality(
            arr, metadata={"modality": "SENTINEL2_MULTISPECTRAL", "band_names": ["B03", "B04", "B08"]}
        )
        assert spec.modality == InputModality.SENTINEL2_MULTISPECTRAL
        assert spec.detection_basis == "metadata"

    def test_shape_heuristic_12ch_suggests_multimodal(self):
        """12 channels → shape heuristic suggests MULTIMODAL_S1_S2, but marked unconfirmed."""
        arr = make_arr(12)
        spec = detect_modality(arr)
        assert spec.modality == InputModality.MULTIMODAL_S1_S2
        assert "unconfirmed" in spec.detection_basis

    def test_shape_heuristic_3ch_suggests_rgb(self):
        """3 channels → shape heuristic suggests RGB, but marked unconfirmed."""
        arr = make_arr(3)
        spec = detect_modality(arr)
        assert spec.modality == InputModality.RGB_OPTICAL
        assert "unconfirmed" in spec.detection_basis

    def test_shape_heuristic_1ch_is_unknown(self):
        """1 channel is ambiguous (SAR/NIR/thermal) → UNKNOWN."""
        arr = make_arr(1)
        spec = detect_modality(arr)
        assert spec.modality == InputModality.UNKNOWN
        assert "unconfirmed" in spec.detection_basis

    def test_shape_heuristic_s2_multispectral(self):
        """4–10 channels → shape heuristic suggests SENTINEL2_MULTISPECTRAL."""
        for c in [4, 6, 10]:
            arr = make_arr(c)
            spec = detect_modality(arr)
            assert spec.modality == InputModality.SENTINEL2_MULTISPECTRAL

    def test_user_hint_invalid_falls_back_to_unknown(self):
        """Invalid user_hint string → UNKNOWN modality."""
        arr = make_arr(6)
        spec = detect_modality(arr, user_hint="MADE_UP_MODALITY")
        assert spec.modality == InputModality.UNKNOWN
        assert spec.detection_basis == "user"

    def test_band_names_s1_s2_mixed(self):
        """Band names with both S2 and SAR → MULTIMODAL_S1_S2."""
        arr = make_arr(12)
        spec = detect_modality(
            arr,
            metadata={"band_names": ["B02", "B03", "B04", "B08", "B05", "B06",
                                      "B07", "B11", "B12", "B8A", "VH", "VV"]},
        )
        assert spec.modality == InputModality.MULTIMODAL_S1_S2
        assert spec.detection_basis == "metadata"

    def test_detection_basis_label_user(self):
        arr = make_arr(3)
        spec = detect_modality(arr, user_hint="RGB_OPTICAL")
        assert spec.basis_label == "User selected"

    def test_detection_basis_label_metadata(self):
        arr = make_arr(3)
        spec = detect_modality(arr, metadata={"modality": "RGB_OPTICAL"})
        assert spec.basis_label == "Detected from metadata"

    def test_detection_basis_label_shape_heuristic(self):
        arr = make_arr(12)
        spec = detect_modality(arr)
        assert "unconfirmed" in spec.basis_label.lower()


# ---------------------------------------------------------------------------
# Pair modality resolution
# ---------------------------------------------------------------------------

class TestPairModality:
    """[SYNTHETIC] Tests for resolve_pair_modality()."""

    def test_same_modality_is_temporal_pair(self):
        spec_a = ModalitySpec(
            modality=InputModality.MULTIMODAL_S1_S2,
            channel_count=12,
            detection_basis="user",
        )
        spec_b = ModalitySpec(
            modality=InputModality.MULTIMODAL_S1_S2,
            channel_count=12,
            detection_basis="user",
        )
        assert resolve_pair_modality(spec_a, spec_b) == InputModality.TEMPORAL_PAIR

    def test_different_modality_is_cross_modal(self):
        spec_a = ModalitySpec(
            modality=InputModality.RGB_OPTICAL,
            channel_count=3,
            detection_basis="user",
        )
        spec_b = ModalitySpec(
            modality=InputModality.SAR_ONLY,
            channel_count=2,
            detection_basis="user",
        )
        assert resolve_pair_modality(spec_a, spec_b) == InputModality.CROSS_MODAL_PAIR


# ---------------------------------------------------------------------------
# SpectralService modality guards
# ---------------------------------------------------------------------------

class TestSpectralGuards:
    """[SYNTHETIC] Tests that spectral service rejects incompatible modalities."""

    def test_spectral_guard_raises_for_sar_only(self):
        """SpectralService must raise ModalityError for SAR_ONLY input."""
        arr = make_arr(2)
        spec = ModalitySpec(
            modality=InputModality.SAR_ONLY,
            channel_count=2,
            detection_basis="user",
        )
        with pytest.raises(ModalityError):
            SpectralService.calculate_ndvi(arr, spec=spec)

    def test_analyze_tile_returns_unavailable_for_sar(self):
        """analyze_tile returns UNAVAILABLE dict (not fabricated values) for SAR."""
        arr = make_arr(2)
        spec = ModalitySpec(
            modality=InputModality.SAR_ONLY,
            channel_count=2,
            detection_basis="user",
        )
        result = SpectralService.analyze_tile(arr, spec=spec)
        assert result.get("modality_error") is True
        assert result["ndvi_stats"]["status"] == "UNAVAILABLE"
        assert result["ndwi_stats"]["status"] == "UNAVAILABLE"
        assert result["ndbi_stats"]["status"] == "UNAVAILABLE"

    def test_analyze_tile_returns_unavailable_for_unknown(self):
        """analyze_tile returns UNAVAILABLE for UNKNOWN modality."""
        arr = make_arr(1)
        spec = ModalitySpec(
            modality=InputModality.UNKNOWN,
            channel_count=1,
            detection_basis="shape_heuristic (unconfirmed)",
        )
        result = SpectralService.analyze_tile(arr, spec=spec)
        assert result.get("modality_error") is True

    def test_spectral_succeeds_for_multimodal(self):
        """SpectralService succeeds for 12-ch multimodal without spec (no guard)."""
        arr = np.random.rand(12, 16, 16).astype(np.float32)
        result = SpectralService.analyze_tile(arr, spec=None)
        # Should produce real float stats, not UNAVAILABLE
        assert "mean" in result["ndvi_stats"]
        assert isinstance(result["ndvi_stats"]["mean"], float)

    def test_bands_absent_in_spec_returns_unavailable(self):
        """When spec.band_names confirms NIR absent, NDVI → UNAVAILABLE."""
        arr = make_arr(3)
        # Only R, G, B — no NIR → NDVI requires B08
        spec = ModalitySpec(
            modality=InputModality.SENTINEL2_MULTISPECTRAL,
            channel_count=3,
            detection_basis="metadata",
            band_names=["B04", "B03", "B02"],
        )
        result = SpectralService.analyze_tile(arr, spec=spec)
        assert result["ndvi_stats"]["status"] == "UNAVAILABLE"
        assert result["ndwi_stats"]["status"] == "UNAVAILABLE"
        assert result["ndbi_stats"]["status"] == "UNAVAILABLE"


# ---------------------------------------------------------------------------
# SARService — no encoder, no padding
# ---------------------------------------------------------------------------

class TestSARService:
    """[SYNTHETIC] Tests that SARService uses only the SAR channels."""

    def test_sar_analyze_1ch(self):
        """Single-channel SAR array produces valid stats."""
        arr = np.random.rand(1, 16, 16).astype(np.float32)
        result = SARService.analyze(arr)
        assert result["num_channels"] == 1
        assert result["polarization_ratio"] is None  # single-pol, no ratio
        assert "scope_note" in result
        assert len(result["channels"]) == 1

    def test_sar_analyze_2ch(self):
        """Dual-channel SAR array produces polarization ratio."""
        arr = np.random.rand(2, 16, 16).astype(np.float32) + 0.1
        result = SARService.analyze(arr)
        assert result["num_channels"] == 2
        assert result["polarization_ratio"] is not None
        assert "mean_ratio" in result["polarization_ratio"]

    def test_sar_analyze_does_not_call_12ch_encoder(self):
        """SARService.analyze() must NOT import or use the 12-ch satellite encoder."""
        import backend.services.sar_service as sar_module
        import inspect
        src = inspect.getsource(sar_module.SARService.analyze)
        # Verify no reference to SatelliteEncoder or VLMService in the method
        assert "SatelliteEncoder" not in src
        assert "VLMService" not in src
        assert "ClassificationService" not in src

    def test_sar_guard_raises_for_rgb(self):
        """SARService must raise ModalityError for RGB_OPTICAL modality."""
        arr = make_arr(3)
        spec = ModalitySpec(
            modality=InputModality.RGB_OPTICAL,
            channel_count=3,
            detection_basis="user",
        )
        with pytest.raises(ModalityError):
            SARService.analyze(arr, spec=spec)

    def test_sar_scope_note_present(self):
        """Every SARService result must include a scope_note explaining limitations."""
        arr = np.random.rand(1, 8, 8).astype(np.float32)
        result = SARService.analyze(arr)
        assert "scope_note" in result
        assert len(result["scope_note"]) > 20  # Non-trivial string

    def test_extract_sar_channels_from_12ch(self):
        """extract_sar_channels returns exactly channels 10 and 11."""
        arr = np.arange(12 * 4 * 4, dtype=np.float32).reshape(12, 4, 4)
        sar = SARService.extract_sar_channels(arr)
        assert sar.shape == (2, 4, 4)
        np.testing.assert_array_equal(sar[0], arr[10])
        np.testing.assert_array_equal(sar[1], arr[11])

    def test_extract_sar_channels_wrong_shape_raises(self):
        """extract_sar_channels raises ValueError for non-12-channel input."""
        arr = make_arr(3)
        with pytest.raises(ValueError):
            SARService.extract_sar_channels(arr)


# ---------------------------------------------------------------------------
# ChangeDetectionService — co-registration caveat
# ---------------------------------------------------------------------------

class TestChangeDetection:
    """[SYNTHETIC] Tests for change detection co-registration enforcement."""

    def test_unregistered_returns_warning_no_pixel_diff(self):
        """co_registered=False → warning present, pixel_change is None."""
        before = make_arr(12)
        after = make_arr(12)
        result = ChangeDetectionService.compare(before, after, co_registered=False)
        assert result["co_registered"] is False
        assert result["warning"] is not None
        assert "co-registr" in result["warning"].lower()
        assert result["pixel_change"] is None

    def test_registered_same_shape_returns_pixel_diff(self):
        """co_registered=True with same shape → pixel_change populated."""
        before = np.zeros((3, 16, 16), dtype=np.float32)
        after = np.ones((3, 16, 16), dtype=np.float32)
        result = ChangeDetectionService.compare(before, after, co_registered=True)
        assert result["co_registered"] is True
        assert result["warning"] is None
        assert result["pixel_change"] is not None
        assert result["pixel_change"]["change_magnitude"]["mean"] > 0

    def test_shape_mismatch_returns_warning(self):
        """Shape mismatch with co_registered=True → warning, no pixel_change."""
        before = make_arr(3, h=16, w=16)
        after = make_arr(3, h=32, w=32)
        result = ChangeDetectionService.compare(before, after, co_registered=True)
        assert result["pixel_change"] is None
        assert result["warning"] is not None

    def test_cross_modal_pair_no_pixel_diff(self):
        """Different modalities → CROSS_MODAL_PAIR, no pixel arithmetic."""
        before = make_arr(3)
        after = make_arr(2)
        spec_before = ModalitySpec(
            modality=InputModality.RGB_OPTICAL, channel_count=3, detection_basis="user"
        )
        spec_after = ModalitySpec(
            modality=InputModality.SAR_ONLY, channel_count=2, detection_basis="user"
        )
        result = ChangeDetectionService.compare(
            before, after,
            co_registered=True,  # even with co_registered=True, cross-modal blocks pixel diff
            spec_before=spec_before,
            spec_after=spec_after,
        )
        assert result["pair_type"] == "CROSS_MODAL_PAIR"
        assert result["pixel_change"] is None
        assert result["warning"] is not None

    def test_temporal_pair_detected_from_same_modality(self):
        """Same modality specs → TEMPORAL_PAIR."""
        before = make_arr(12)
        after = make_arr(12)
        spec = ModalitySpec(
            modality=InputModality.MULTIMODAL_S1_S2, channel_count=12, detection_basis="user"
        )
        result = ChangeDetectionService.compare(
            before, after, co_registered=False, spec_before=spec, spec_after=spec
        )
        assert result["pair_type"] == "TEMPORAL_PAIR"

    def test_per_image_stats_always_present(self):
        """image_before and image_after stats are always returned."""
        before = np.random.rand(4, 8, 8).astype(np.float32)
        after = np.random.rand(4, 8, 8).astype(np.float32)
        result = ChangeDetectionService.compare(before, after, co_registered=False)
        assert "channels" in result["image_before"]
        assert "channels" in result["image_after"]
        assert len(result["image_before"]["channels"]) == 4


# ---------------------------------------------------------------------------
# require_analysis helper
# ---------------------------------------------------------------------------

class TestRequireAnalysis:
    """[SYNTHETIC] Tests for the require_analysis() guard helper."""

    def test_raises_for_unavailable_analysis(self):
        spec = ModalitySpec(
            modality=InputModality.RGB_OPTICAL,
            channel_count=3,
            detection_basis="user",
        )
        with pytest.raises(ModalityError):
            require_analysis(spec, "sar_backscatter")

    def test_passes_for_available_analysis(self):
        spec = ModalitySpec(
            modality=InputModality.RGB_OPTICAL,
            channel_count=3,
            detection_basis="user",
        )
        # Should not raise
        require_analysis(spec, "rgb_composite")
        require_analysis(spec, "vlm_query")
