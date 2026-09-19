"""
satquery/services/rgb_service.py
─────────────────────────────────
Dedicated separate RGB vision path for 3-channel optical and aerial imagery.
Never passes RGB through the 12-channel BigEarthNet S1+S2 encoder.
"""

from typing import Dict, Any, List, Union, Optional
import numpy as np
import torch
import torch.nn as nn
from backend.data.modality import standardize_image_input


class RGBVisionService:
    """
    Dedicated 3-channel RGB Vision Service.
    Analyzes scene characteristics, optical features, and land-cover categories
    without faking channels or invoking 12-channel multispectral encoders.
    """

    RGB_CLASSES = [
        "Forest & Vegetation",
        "Water Body",
        "Urban & Built-up Fabric",
        "Agricultural Land",
        "Roads & Infrastructure",
        "Bare Soil & Sand",
    ]

    @classmethod
    def analyze_scene(cls, image_data: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Analyze RGB imagery and extract dominant scene characteristics and features.
        """
        arr = standardize_image_input(image_data)
        if arr.shape[0] != 3:
            raise ValueError(f"RGBVisionService requires 3-channel optical input, got shape {arr.shape}")

        r, g, b = arr[0], arr[1], arr[2]
        h, w = r.shape
        total_pixels = float(h * w)

        # 1. Color and spectral signatures in RGB space
        # Visible green excess (green minus red) for vegetation
        veg_mask = (g > r) & (g > b) & (g > 0.15)
        veg_pct = float(np.sum(veg_mask)) / total_pixels * 100.0

        # Water: high blue/green relative to red, lower overall reflectance
        water_mask = (b >= r) & (g >= r) & ((r + g + b) < 0.9) & (b > 0.08)
        water_pct = float(np.sum(water_mask)) / total_pixels * 100.0

        # Urban / Built-up: high variance / edge density, neutral grey tones
        grey_diff = np.maximum(np.abs(r - g), np.abs(g - b))
        brightness = (r + g + b) / 3.0
        urban_mask = (grey_diff < 0.08) & (brightness > 0.25) & (brightness < 0.85)
        urban_pct = float(np.sum(urban_mask)) / total_pixels * 100.0

        # Agriculture: moderate vegetation with warm/brown soil mix
        agri_mask = ((r > 0.3) & (g > 0.25) & (b < 0.3) & ~veg_mask) | (veg_mask & (brightness > 0.45))
        agri_pct = float(np.sum(agri_mask)) / total_pixels * 100.0

        # Roads: high contrast dark/light linear patterns
        dy, dx = np.gradient(brightness)
        grad_mag = np.sqrt(dx**2 + dy**2)
        road_mask = (grad_mag > 0.12) & (grey_diff < 0.1)
        road_pct = float(np.sum(road_mask)) / total_pixels * 100.0

        # Bare soil / sand
        sand_mask = (r > 0.4) & (g > 0.35) & (b > 0.2) & (r > g) & (g > b)
        sand_pct = float(np.sum(sand_mask)) / total_pixels * 100.0

        # Compute confidence and ranking
        scores = {
            "Forest & Vegetation": min(1.0, veg_pct / 60.0 + 0.05),
            "Water Body": min(1.0, water_pct / 50.0 + 0.05),
            "Urban & Built-up Fabric": min(1.0, urban_pct / 45.0 + 0.05),
            "Agricultural Land": min(1.0, agri_pct / 50.0 + 0.05),
            "Roads & Infrastructure": min(1.0, road_pct / 20.0 + 0.05),
            "Bare Soil & Sand": min(1.0, sand_pct / 40.0 + 0.05),
        }

        # Normalize relative probabilities
        total_score = sum(scores.values()) or 1.0
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_class = ranked[0][0]
        primary_conf = round(float(ranked[0][1] / total_score), 4)

        # Scene summary description
        desc_parts = []
        if veg_pct > 20:
            desc_parts.append(f"significant vegetative canopy ({veg_pct:.1f}% coverage)")
        if water_pct > 10:
            desc_parts.append(f"hydrological surfaces / water bodies ({water_pct:.1f}% coverage)")
        if urban_pct > 15:
            desc_parts.append(f"built-up impervious structures ({urban_pct:.1f}% coverage)")
        if agri_pct > 15:
            desc_parts.append(f"cultivated fields / agriculture ({agri_pct:.1f}% coverage)")
        if road_pct > 5:
            desc_parts.append(f"transportation / road corridors ({road_pct:.1f}% coverage)")

        scene_desc = (
            f"Standard 3-band RGB optical analysis reveals a landscape characterized by "
            + (", ".join(desc_parts) if desc_parts else "mixed land cover")
            + f", with primary land category: {primary_class}."
        )

        return {
            "modality": "RGB_OPTICAL",
            "primary_class": primary_class,
            "confidence": primary_conf,
            "top_predictions": [{"class_name": k, "confidence": round(float(v / total_score), 4)} for k, v in ranked[:5]],
            "feature_coverages": {
                "vegetation_percent": round(veg_pct, 2),
                "water_percent": round(water_pct, 2),
                "urban_percent": round(urban_pct, 2),
                "agriculture_percent": round(agri_pct, 2),
                "roads_percent": round(road_pct, 2),
                "bare_soil_percent": round(sand_pct, 2),
            },
            "scene_description": scene_desc,
        }
