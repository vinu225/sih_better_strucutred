"""
satquery/services/rgb_service.py
─────────────────────────────────
Dedicated separate RGB vision path for 3-channel optical and aerial imagery.
Never passes RGB through the 12-channel BigEarthNet S1+S2 encoder.
Returns structured coverage estimates, neutral scene descriptions, and localized regions.
"""

from typing import Dict, Any, List, Union, Optional
import numpy as np
import torch
from backend.data.modality import standardize_image_input
from backend.services.grounding_service import GroundingService


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
    def estimate_coverage(cls, image_data: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Estimate surface cover percentages using explicit 3-channel color heuristics.
        """
        arr = standardize_image_input(image_data)
        if arr.shape[0] != 3:
            raise ValueError(f"RGBVisionService requires 3-channel optical input, got shape {arr.shape}")

        r, g, b = arr[0].astype(np.float32), arr[1].astype(np.float32), arr[2].astype(np.float32)
        h, w = r.shape
        total_pixels = float(h * w)

        # 1. Vegetation: Green excess over red and blue
        veg_mask = (g > r * 1.02) & (g > b * 1.02) & (g > 0.12)
        veg_pct = round(float(np.sum(veg_mask)) / total_pixels * 100.0, 2)

        # 2. Water: Blue/Green dominant, low overall brightness
        brightness = (r + g + b) / 3.0
        water_mask = (b >= r) & (g >= r * 0.9) & (brightness < 0.45) & (b > 0.06)
        water_pct = round(float(np.sum(water_mask)) / total_pixels * 100.0, 2)

        # 3. Built-up / Urban: Neutral grey tones with edge/texture gradient
        grey_diff = np.maximum(np.abs(r - g), np.abs(g - b))
        dy, dx = np.gradient(brightness)
        edges = np.sqrt(dx**2 + dy**2)
        urban_mask = (grey_diff < 0.12) & (brightness > 0.22) & (brightness < 0.88) & (edges > 0.035)
        urban_pct = round(float(np.sum(urban_mask)) / total_pixels * 100.0, 2)

        # 4. Agriculture: Warm soil-crop blend or bright uniform vegetation
        agri_mask = ((r > 0.28) & (g > 0.24) & (b < 0.32) & ~veg_mask) | (veg_mask & (brightness > 0.42))
        agri_pct = round(float(np.sum(agri_mask)) / total_pixels * 100.0, 2)

        # 5. Roads & Infrastructure: High contrast linear segments
        road_mask = (edges > 0.08) & (grey_diff < 0.10) & (brightness > 0.15) & (brightness < 0.80)
        road_pct = round(float(np.sum(road_mask)) / total_pixels * 100.0, 2)

        # 6. Bare soil / sand: High red/yellow reflectance
        sand_mask = (r > 0.38) & (g > 0.32) & (b > 0.18) & (r >= g) & (g >= b)
        sand_pct = round(float(np.sum(sand_mask)) / total_pixels * 100.0, 2)

        return {
            "vegetation": veg_pct,
            "agriculture": agri_pct,
            "water": water_pct,
            "built_up": urban_pct,
            "roads": road_pct,
            "bare": sand_pct,
            "method": "RGB color heuristic",
            "sensor_note": "Estimated from 3-band visible color heuristics; physical NDVI/NDWI require Sentinel-2 NIR/SWIR bands.",
        }

    @staticmethod
    def get_magnitude(pct: float) -> str:
        """Categorize coverage percentage into fixed magnitude descriptor."""
        if pct >= 50.0:
            return "dominant"
        if pct >= 20.0:
            return "moderate"
        if pct >= 2.0:
            return "small"
        return "negligible"

    @classmethod
    def get_ranked_classes(cls, cov: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return coverage classes sorted in descending order of percentage."""
        class_meta = [
            ("vegetation", "Vegetation"),
            ("agriculture", "Agricultural land"),
            ("water", "Water bodies"),
            ("built_up", "Built-up areas"),
            ("roads", "Roads & infrastructure"),
            ("bare", "Bare soil"),
        ]
        items = []
        for key, label in class_meta:
            pct = round(float(cov.get(key, 0.0)), 1)
            items.append({
                "key": key,
                "label": label,
                "percentage": pct,
                "magnitude": cls.get_magnitude(pct),
            })
        items.sort(key=lambda x: x["percentage"], reverse=True)
        return items

    @classmethod
    def format_coverage_answer(
        cls,
        cov: Dict[str, Any],
        target_key: str,
        target_label: str,
        variation: int = 0
    ) -> str:
        """
        Generate a 2-3 sentence answer for a specific coverage inquiry on RGB imagery.
        """
        ranked = cls.get_ranked_classes(cov)
        target_item = next((item for item in ranked if item["key"] == target_key), None)
        pct = target_item["percentage"] if target_item else round(float(cov.get(target_key, 0.0)), 1)
        mag = cls.get_magnitude(pct)

        top_item = ranked[0]
        second_item = ranked[1] if len(ranked) > 1 else ranked[0]

        # Sentence 1: Direct answer to asked class
        if pct >= 1.0:
            s1 = f"{target_label} coverage is estimated at {pct:.1f}% across this optical tile."
        else:
            s1 = f"{target_label} is barely present across this optical tile, with an estimated coverage of {pct:.1f}%."

        # Sentence 2: Context from rank and second/dominant class
        if target_key == top_item["key"]:
            if variation == 0:
                s2 = f"It constitutes the dominant surface category in the scene, ahead of {second_item['label'].lower()} at {second_item['percentage']:.1f}%."
            else:
                s2 = f"It ranks as the primary land cover across the image, followed by {second_item['label'].lower()} ({second_item['percentage']:.1f}%)."
        else:
            if variation == 0:
                s2 = f"It ranks as a {mag} component behind {top_item['label'].lower()} ({top_item['percentage']:.1f}%), which is the primary land cover."
            else:
                s2 = f"It represents a {mag} share of the tile, with {top_item['label'].lower()} ({top_item['percentage']:.1f}%) being the dominant surface type."

        # Sentence 3: Practical takeaway / next inquiry
        s3 = "You can ask for the full land-cover breakdown or upload a 12-channel multispectral tile for physical spectral indices."

        note = "*(Note: Estimated from 3-band visible RGB color heuristics. Accurate physical spectral indices such as NDVI, NDWI, or NDBI require Sentinel-2 Near-Infrared and Shortwave Infrared bands.)*"
        return f"{s1} {s2} {s3}\n\n{note}"

    @classmethod
    def format_describe_answer(cls, cov: Dict[str, Any], variation: int = 0) -> str:
        """
        Generate a 2-3 sentence neutral scene description for RGB imagery.
        """
        ranked = cls.get_ranked_classes(cov)
        r0, r1, r2 = ranked[0], ranked[1], ranked[2]

        # Sentence 1: Top 2-3 classes with percentages
        if r2["percentage"] >= 1.0:
            s1 = f"This optical scene is characterized by {r0['label'].lower()} ({r0['percentage']:.1f}%), {r1['label'].lower()} ({r1['percentage']:.1f}%), and {r2['label'].lower()} ({r2['percentage']:.1f}%)."
        else:
            s1 = f"This optical scene is predominantly composed of {r0['label'].lower()} ({r0['percentage']:.1f}%) and {r1['label'].lower()} ({r1['percentage']:.1f}%)."

        # Sentence 2: Magnitude overview
        if variation == 0:
            s2 = f"{r0['label']} forms the {r0['magnitude']} land cover, while other surface types make up the remainder of the terrain."
        else:
            s2 = f"{r0['label']} represents the primary surface feature across the tile, accompanied by moderate {r1['label'].lower()}."

        # Sentence 3: Suggested next question
        s3 = "You can query specific surface category percentages or request spatial localization to inspect candidate features."

        note = "*(Note: Estimated from visible RGB color heuristics. True multispectral indices like NDVI/NDWI require NIR/SWIR bands.)*"
        return f"{s1} {s2} {s3}\n\n{note}"

    @classmethod
    def format_grounding_answer(
        cls,
        cov: Dict[str, Any],
        target_concept: str,
        target_key: str,
        target_label: str,
        boxes_count: int = 0,
        variation: int = 0
    ) -> str:
        """
        Generate a 2-3 sentence answer for grounding/localization on RGB imagery.
        """
        ranked = cls.get_ranked_classes(cov)
        target_item = next((item for item in ranked if item["key"] == target_key), None)
        pct = target_item["percentage"] if target_item else round(float(cov.get(target_key, 0.0)), 1)
        mag = cls.get_magnitude(pct)
        top_item = ranked[0]

        # Sentence 1: Clear statement that individual locations cannot be reliably marked
        s1 = f"Individual bounding box locations for '{target_concept}' cannot be reliably marked on standard 3-channel RGB optical imagery."

        # Sentence 2: Overall coverage share and ranking context
        if target_key == top_item["key"]:
            if variation == 0:
                s2 = f"Overall {target_label.lower()} coverage is estimated at {pct:.1f}%, making it the dominant surface category across the tile."
            else:
                s2 = f"{target_label} is the highest-ranked land category in the scene, accounting for an estimated {pct:.1f}%."
        else:
            if variation == 0:
                s2 = f"Overall {target_label.lower()} coverage is estimated at {pct:.1f}%, ranking as a {mag} component behind {top_item['label'].lower()} ({top_item['percentage']:.1f}%)."
            else:
                s2 = f"{target_label} accounts for a {mag} share of {pct:.1f}%, while {top_item['label'].lower()} ({top_item['percentage']:.1f}%) is the primary surface type."

        # Sentence 3: Practical recommendation
        s3 = "You can request percentage coverage for any class or upload a Sentinel-2 multispectral tile for spatial spectral segmentation."

        note = "*(Note: Estimated from 3-band visible color heuristics. Physical feature isolation requires Sentinel-2 multispectral bands.)*"
        return f"{s1} {s2} {s3}\n\n{note}"

    @classmethod
    def format_challenge_answer(
        cls,
        cov: Dict[str, Any],
        target_key: str,
        target_label: str,
        variation: int = 0
    ) -> str:
        """
        Generate a 2-3 sentence answer for a challenge/doubt query on RGB imagery.
        """
        ranked = cls.get_ranked_classes(cov)
        target_item = next((item for item in ranked if item["key"] == target_key), None)
        pct = target_item["percentage"] if target_item else round(float(cov.get(target_key, 0.0)), 1)
        top_item = ranked[0]
        second_item = ranked[1] if len(ranked) > 1 else ranked[0]

        # Sentence 1: Direct confirm or correct
        if pct >= 1.0:
            s1 = f"No, that is incorrect, as {target_label.lower()} accounts for an estimated {pct:.1f}% of the visible surface area."
            if target_key == top_item["key"]:
                if variation == 0:
                    s2 = f"{target_label} is the dominant surface category in this image, ahead of {second_item['label'].lower()} at {second_item['percentage']:.1f}%."
                else:
                    s2 = f"{target_label} ranks first among all detected land cover types, with {second_item['label'].lower()} ({second_item['percentage']:.1f}%) in second."
            else:
                if variation == 0:
                    s2 = f"{target_label} is present as a {cls.get_magnitude(pct)} feature, while {top_item['label'].lower()} ({top_item['percentage']:.1f}%) is the primary land cover."
                else:
                    s2 = f"It ranks behind {top_item['label'].lower()} ({top_item['percentage']:.1f}%), which forms the dominant feature across the tile."
        else:
            s1 = f"Yes, that is consistent with the image, as {target_label.lower()} is barely present at an estimated {pct:.1f}%."
            if variation == 0:
                s2 = f"Instead, {top_item['label'].lower()} is the dominant category across the scene at {top_item['percentage']:.1f}%."
            else:
                s2 = f"The landscape is primarily characterized by {top_item['label'].lower()} ({top_item['percentage']:.1f}%)."

        # Sentence 3: Next step / takeaway
        s3 = "You can query any other surface category or request a full percentage breakdown."

        note = "*(Note: Derived from 3-band visible color heuristics; physical validation requires Sentinel-2 NIR/SWIR multispectral bands.)*"
        return f"{s1} {s2} {s3}\n\n{note}"

    @classmethod
    def describe_scene(cls, image_data: Union[np.ndarray, torch.Tensor]) -> str:
        """
        Produce a 2-3 sentence neutral scene description with trailing note.
        """
        cov = cls.estimate_coverage(image_data)
        return cls.format_describe_answer(cov)

    @classmethod
    def locate(cls, image_data: Union[np.ndarray, torch.Tensor], target: str) -> List[Dict[str, Any]]:
        """
        Localize specific targets/features on RGB image, returning bounding boxes.
        """
        res = GroundingService.ground_concept(image_data, target)
        boxes = []
        for b in res.get("boxes", []):
            boxes.append({
                "label": b.get("label", target),
                "box": b.get("box_2d", []),
                "confidence": b.get("confidence", 0.7),
            })
        return boxes

    @classmethod
    def analyze_scene(cls, image_data: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Comprehensive backward-compatible RGB scene analysis returning structured characteristics.
        """
        cov = cls.estimate_coverage(image_data)
        desc = cls.describe_scene(image_data)

        scores = {
            "Forest & Vegetation": min(1.0, cov["vegetation"] / 60.0 + 0.05),
            "Water Body": min(1.0, cov["water"] / 50.0 + 0.05),
            "Urban & Built-up Fabric": min(1.0, cov["built_up"] / 45.0 + 0.05),
            "Agricultural Land": min(1.0, cov["agriculture"] / 50.0 + 0.05),
            "Roads & Infrastructure": min(1.0, cov["roads"] / 20.0 + 0.05),
            "Bare Soil & Sand": min(1.0, cov["bare"] / 40.0 + 0.05),
        }

        total_score = sum(scores.values()) or 1.0
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_class = ranked[0][0]
        primary_conf = round(float(ranked[0][1] / total_score), 4)

        return {
            "modality": "RGB_OPTICAL",
            "primary_class": primary_class,
            "confidence": primary_conf,
            "top_predictions": [{"class_name": k, "confidence": round(float(v / total_score), 4)} for k, v in ranked[:5]],
            "feature_coverages": {
                "vegetation_percent": cov["vegetation"],
                "water_percent": cov["water"],
                "urban_percent": cov["built_up"],
                "agriculture_percent": cov["agriculture"],
                "roads_percent": cov["roads"],
                "bare_soil_percent": cov["bare"],
            },
            "coverage_estimates": cov,
            "scene_description": desc,
        }
