"""
satquery/agent/tools.py
───────────────────────
Model & Tool Registry with capability metadata for SatQuery AI Agent.
Enables dynamic specialist selection based on modality, band availability, and query intent.
"""

from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import numpy as np
import torch

from backend.config import BAND_NAMES, BIGEARTHNET_19_CLASSES
from backend.data.modality import InputModality, ModalitySpec, standardize_image_input
from backend.services.spectral_service import SpectralService
from backend.services.classification_service import ClassificationService
from backend.services.vlm_service import VLMService
from backend.services.sar_service import SARService
from backend.services.change_detection_service import ChangeDetectionService
from backend.services.rgb_service import RGBVisionService
from backend.services.grounding_service import GroundingService
from backend.services.cross_modal_service import CrossModalService


class Tool:
    """Agent Tool specification with capability metadata."""
    def __init__(
        self,
        name: str,
        task_type: str,
        description: str,
        func: Callable,
        supported_modalities: Optional[List[InputModality]] = None,
        required_bands: Optional[List[str]] = None,
        supports_pairs: bool = False,
    ):
        self.name = name
        self.task_type = task_type
        self.description = description
        self.func = func
        self.supported_modalities = supported_modalities or []
        self.required_bands = required_bands or []
        self.supports_pairs = supports_pairs

    def check_compatibility(
        self,
        modality: InputModality,
        image_count: int = 1,
        available_bands: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Verify if this tool can process the given input."""
        if self.supports_pairs and image_count < 2:
            return False, f"Tool '{self.name}' requires 2 images, but received {image_count}."
        if not self.supports_pairs and image_count > 1:
            return False, f"Tool '{self.name}' operates on a single image, but received {image_count} images."

        if self.supported_modalities and modality not in self.supported_modalities:
            return False, (
                f"Modality '{modality.value}' is incompatible with '{self.name}'. "
                f"Supported: {[m.value for m in self.supported_modalities]}."
            )

        if self.required_bands and available_bands:
            missing = [b for b in self.required_bands if b not in available_bands]
            if missing:
                return False, f"Missing required bands {missing} for '{self.name}'."

        return True, "Compatible"

    def execute(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Specialist Tool Implementations
# ---------------------------------------------------------------------------

def spectral_analysis_tool(data: Union[torch.Tensor, np.ndarray], **kwargs) -> Dict[str, Any]:
    """Calculate NDVI, NDWI, NDBI from multispectral data."""
    arr = standardize_image_input(data)
    res = SpectralService.analyze_tile(arr)
    return {
        "ndvi_mean": res["ndvi_stats"].get("mean"),
        "ndwi_mean": res["ndwi_stats"].get("mean"),
        "ndbi_mean": res["ndbi_stats"].get("mean"),
        "coverage": res.get("coverage_estimates", {}),
        "assessment": res.get("assessment", ""),
    }


def landcover_classification_tool(data: Union[torch.Tensor, np.ndarray], **kwargs) -> Dict[str, Any]:
    """Classify land cover. Routes S1+S2 to real MobileViT-s, RGB to RGB classifier."""
    arr = standardize_image_input(data)
    if arr.shape[0] == 3:
        # Separate RGB path
        rgb_res = RGBVisionService.analyze_scene(arr)
        return {
            "primary_class": rgb_res["primary_class"],
            "confidence": rgb_res["confidence"],
            "top_5": rgb_res["top_predictions"],
            "detected_classes": [p["class_name"] for p in rgb_res["top_predictions"]],
            "model_used": "RGBVisionClassifier",
        }
    else:
        # Multispectral / S1+S2 real MobileViT-s path
        res = ClassificationService.classify_tile(arr, top_k=5)
        return {
            "primary_class": res["primary_class"],
            "confidence": res["primary_confidence"],
            "top_5": res["top_predictions"],
            "detected_classes": [c["class_name"] for c in res["detected_classes"]],
            "model_used": "BigEarthNet_MobileViT-s",
        }


def rgb_scene_analysis_tool(data: Union[torch.Tensor, np.ndarray], **kwargs) -> Dict[str, Any]:
    """Analyze 3-channel RGB image without touching 12-channel encoder."""
    arr = standardize_image_input(data)
    return RGBVisionService.analyze_scene(arr)


def grounding_localization_tool(data: Union[torch.Tensor, np.ndarray], target_concept: str = "water", **kwargs) -> Dict[str, Any]:
    """Locate objects/regions, return bounding boxes and visual overlay."""
    return GroundingService.ground_concept(data, target_concept)


def change_detection_tool(
    before_data: Union[torch.Tensor, np.ndarray],
    after_data: Union[torch.Tensor, np.ndarray],
    co_registered: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Compare temporal pair images and return difference metrics."""
    arr_b = standardize_image_input(before_data)
    arr_a = standardize_image_input(after_data)
    return ChangeDetectionService.compare(arr_b, arr_a, co_registered=co_registered)


def cross_modal_tool(
    optical_data: Union[torch.Tensor, np.ndarray],
    sar_data: Union[torch.Tensor, np.ndarray],
    query: str = "Optical and SAR joint analysis",
    **kwargs
) -> Dict[str, Any]:
    """Joint reasoning combining optical reflectance and SAR radar backscatter."""
    return CrossModalService.analyze_pair(optical_data, sar_data, query=query)


def sar_analysis_tool(data: Union[torch.Tensor, np.ndarray], **kwargs) -> Dict[str, Any]:
    """Compute SAR backscatter intensity statistics and polarization ratio."""
    arr = standardize_image_input(data)
    if arr.shape[0] == 12:
        # Extract channels 10 and 11 (VH, VV)
        arr = SARService.extract_sar_channels(arr)
    return SARService.analyze(arr)


def vlm_qa_tool(data: Union[torch.Tensor, np.ndarray], question: str, **kwargs) -> Dict[str, Any]:
    """Answer natural language visual questions using SatQueryVLM."""
    arr = standardize_image_input(data)
    # If 3-channel RGB, enrich with RGB analysis rather than faking 12-ch
    if arr.shape[0] == 3:
        desc = RGBVisionService.describe_scene(arr)
        return {
            "question": question,
            "answer": desc,
            "confidence": 0.85,
            "primary_landcover": "Optical RGB Scene",
        }
    return VLMService.query(arr, question)


def band_metadata_tool(band_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Retrieve technical specs for Sentinel-2 optical bands and Sentinel-1 SAR channels."""
    catalog = {
        "B02": {"type": "Optical (Sentinel-2)", "wavelength": "490 nm", "resolution": "10m", "use": "Blue - vegetation, bathymetry, atmospheric scattering"},
        "B03": {"type": "Optical (Sentinel-2)", "wavelength": "560 nm", "resolution": "10m", "use": "Green - vegetation vigor, green canopy peak"},
        "B04": {"type": "Optical (Sentinel-2)", "wavelength": "665 nm", "resolution": "10m", "use": "Red - chlorophyll absorption, NDVI calculation"},
        "B08": {"type": "Optical (Sentinel-2)", "wavelength": "842 nm", "resolution": "10m", "use": "Near Infrared (NIR) - biomass, cell structure, NDVI"},
        "B05": {"type": "Optical (Sentinel-2)", "wavelength": "705 nm", "resolution": "20m", "use": "Vegetation Red Edge 1 - leaf chlorophyll"},
        "B06": {"type": "Optical (Sentinel-2)", "wavelength": "740 nm", "resolution": "20m", "use": "Vegetation Red Edge 2 - leaf area index (LAI)"},
        "B07": {"type": "Optical (Sentinel-2)", "wavelength": "783 nm", "resolution": "20m", "use": "Vegetation Red Edge 3 - canopy nitrogen status"},
        "B11": {"type": "Optical (Sentinel-2)", "wavelength": "1610 nm", "resolution": "20m", "use": "Shortwave Infrared 1 (SWIR-1) - soil moisture, NDBI"},
        "B12": {"type": "Optical (Sentinel-2)", "wavelength": "2190 nm", "resolution": "20m", "use": "Shortwave Infrared 2 (SWIR-2) - geology, burn scars"},
        "B8A": {"type": "Optical (Sentinel-2)", "wavelength": "865 nm", "resolution": "20m", "use": "Narrow NIR - atmospheric water correction"},
        "VH":  {"type": "Synthetic Aperture Radar (Sentinel-1)", "polarization": "Cross-pol (VH)", "frequency": "C-band (5.405 GHz)", "use": "Volumetric radar backscatter, forest canopy roughness, vegetation structure"},
        "VV":  {"type": "Synthetic Aperture Radar (Sentinel-1)", "polarization": "Co-pol (VV)", "frequency": "C-band (5.405 GHz)", "use": "Surface scattering, soil moisture, water boundary mapping"},
    }
    if band_name:
        b_key = band_name.upper()
        if b_key in catalog:
            return {b_key: catalog[b_key]}
        return {"error": f"Channel '{band_name}' not found. Available: {list(catalog.keys())}"}
    return catalog


# ---------------------------------------------------------------------------
# Tool Registry Class
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Central Registry with full capability metadata for all SatQuery specialists.
    """
    def __init__(self):
        self.tools: Dict[str, Tool] = {
            "rgb_scene_analysis": Tool(
                name="rgb_scene_analysis",
                task_type="rgb_scene_analysis",
                description="Performs color, textural, and feature analysis on standard 3-channel optical RGB imagery.",
                func=rgb_scene_analysis_tool,
                supported_modalities=[InputModality.RGB_OPTICAL],
            ),
            "landcover_classification": Tool(
                name="landcover_classification",
                task_type="landcover_classification",
                description="Predicts dominant land-cover categories using BigEarthNet MobileViT-s (for S1+S2) or RGB classifier.",
                func=landcover_classification_tool,
                supported_modalities=[
                    InputModality.MULTIMODAL_S1_S2,
                    InputModality.SENTINEL2_MULTISPECTRAL,
                    InputModality.RGB_OPTICAL,
                ],
            ),
            "grounding_localization": Tool(
                name="grounding_localization",
                task_type="grounding",
                description="Localizes arbitrary concepts/features (water, trees, buildings, roads, agriculture) and returns bounding boxes + visual overlay.",
                func=grounding_localization_tool,
                supported_modalities=[
                    InputModality.RGB_OPTICAL,
                    InputModality.MULTIMODAL_S1_S2,
                    InputModality.SENTINEL2_MULTISPECTRAL,
                ],
            ),
            "change_detection": Tool(
                name="change_detection",
                task_type="change_detection",
                description="Computes before/after temporal change metrics and spatial difference maps for image pairs.",
                func=change_detection_tool,
                supported_modalities=[InputModality.TEMPORAL_PAIR, InputModality.MULTIMODAL_S1_S2, InputModality.RGB_OPTICAL],
                supports_pairs=True,
            ),
            "cross_modal_analysis": Tool(
                name="cross_modal_analysis",
                task_type="cross_modal_analysis",
                description="Joint reasoning combining optical surface reflectance and SAR microwave radar backscatter.",
                func=cross_modal_tool,
                supported_modalities=[InputModality.CROSS_MODAL_PAIR],
                supports_pairs=True,
            ),
            "sar_analysis": Tool(
                name="sar_analysis",
                task_type="sar_analysis",
                description="Calculates Sentinel-1 C-band SAR backscatter statistics (mean, std, min, max, dB) and polarization ratios (VH/VV).",
                func=sar_analysis_tool,
                supported_modalities=[InputModality.SAR_ONLY, InputModality.MULTIMODAL_S1_S2],
            ),
            "spectral_analysis": Tool(
                name="spectral_analysis",
                task_type="spectral_analysis",
                description="Computes verified physical remote sensing indices (NDVI for vegetation, NDWI for water, NDBI for built-up) from multispectral bands.",
                func=spectral_analysis_tool,
                supported_modalities=[InputModality.MULTIMODAL_S1_S2, InputModality.SENTINEL2_MULTISPECTRAL],
                required_bands=["B04", "B08"],
            ),
            "visual_qa": Tool(
                name="visual_qa",
                task_type="vqa",
                description="Answers open-ended natural language reasoning and scene assessment questions via multimodal VLM.",
                func=vlm_qa_tool,
                supported_modalities=[
                    InputModality.MULTIMODAL_S1_S2,
                    InputModality.SENTINEL2_MULTISPECTRAL,
                    InputModality.RGB_OPTICAL,
                    InputModality.SAR_ONLY,
                ],
            ),
            "band_metadata": Tool(
                name="band_metadata",
                task_type="band_metadata",
                description="Provides technical wavelengths, spatial resolutions, and sensors for Sentinel-1 and Sentinel-2 bands.",
                func=band_metadata_tool,
            ),
        }

    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "task_type": t.task_type,
                "description": t.description,
                "supported_modalities": [m.value for m in t.supported_modalities],
                "supports_pairs": t.supports_pairs,
            }
            for t in self.tools.values()
        ]


# Export default instances for backward compatibility
SpectralAnalysisTool = Tool("spectral_analysis", "spectral_analysis", "NDVI/NDWI/NDBI analysis", spectral_analysis_tool)
LandCoverClassificationTool = Tool("landcover_classification", "landcover_classification", "Land-cover classifier", landcover_classification_tool)
VisualQATool = Tool("visual_qa", "vqa", "VLM Visual QA", vlm_qa_tool)
VisualQuestionAnsweringTool = VisualQATool
BandMetadataTool = Tool("band_metadata", "band_metadata", "Band specs", band_metadata_tool)
