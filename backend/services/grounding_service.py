"""
backend/services/grounding_service.py
──────────────────────────────────────
Spatial grounding and object/region localization for Earth Observation imagery.
Detects arbitrary concepts (water, trees, buildings, roads, agriculture, regions),
computes bounding boxes [ymin, xmin, ymax, xmax], and renders visual overlay evidence.
Includes fail-safe fallback when OpenCV (cv2) is not installed.
"""

from typing import Dict, Any, List, Union, Tuple, Optional
import io
import base64
import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from backend.data.modality import standardize_image_input, to_rgb_image, detect_modality, InputModality


class GroundingService:
    """
    Spatial grounding and localization service.
    Extracts explicit bounding boxes, connected region masks, and renders visual overlay evidence.
    """

    COLOR_MAP = {
        "water": (0, 160, 255),       # Cyan / Blue
        "vegetation": (46, 204, 113),  # Emerald Green
        "forest": (39, 174, 96),      # Dark Forest Green
        "building": (231, 76, 60),     # Red / Coral
        "urban": (230, 126, 34),       # Orange
        "road": (241, 196, 15),       # Amber / Yellow
        "agriculture": (155, 89, 182), # Purple / Violet
        "field": (142, 68, 173),
        "soil": (211, 84, 0),         # Ochre
        "default": (52, 152, 219),
    }

    @classmethod
    def ground_concept(
        cls,
        image_data: Union[np.ndarray, Image.Image, str],
        target_concept: str,
        confidence_threshold: float = 0.35,
        max_boxes: int = 12,
    ) -> Dict[str, Any]:
        """
        Localize regions corresponding to target_concept in the satellite/aerial image.
        Returns explicit bounding boxes and a base64-encoded visual evidence overlay.
        """
        arr = standardize_image_input(image_data)
        modality_spec = detect_modality(arr)
        c, h, w = arr.shape
        target_lower = target_concept.lower().strip()

        # 1. Compute binary activation map according to modality and target concept
        mask, feature_name, base_color = cls._generate_activation_mask(arr, modality_spec.modality, target_lower)

        # 2. Extract bounding boxes using contour / connected-component analysis
        u8_mask = (mask * 255).astype(np.uint8)
        total_pixels = float(h * w)

        if _HAS_CV2:
            # Morphological clean up using OpenCV
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned_mask = cv2.morphologyEx(u8_mask, cv2.MORPH_OPEN, kernel)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

            boxes = []
            min_area = max(16, int(total_pixels * 0.002))

            for i, cnt in enumerate(sorted_contours[:max_boxes]):
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                norm_box = [
                    round(float(y) / h, 4),
                    round(float(x) / w, 4),
                    round(float(y + bh) / h, 4),
                    round(float(x + bw) / w, 4),
                ]

                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area / hull_area) if hull_area > 0 else 0.5
                box_conf = round(float(np.clip(0.65 + 0.3 * solidity + 0.05 * min(1.0, area / 500.0), 0.5, 0.98)), 3)

                boxes.append({
                    "id": i + 1,
                    "label": f"{feature_name}_{i+1}",
                    "box_2d": norm_box,
                    "pixel_coords": [y, x, y + bh, x + bw],
                    "confidence": box_conf,
                    "area_pixels": int(area),
                })
        else:
            # Pure NumPy / PIL fallback when cv2 is not available
            cleaned_mask = u8_mask
            boxes = cls._fallback_find_boxes(cleaned_mask, feature_name, h, w, max_boxes)

        matched_pixels = float(np.sum(cleaned_mask > 0))
        coverage_pct = round((matched_pixels / total_pixels) * 100.0, 2)

        # 3. Formulate Visual Evidence Overlay
        base_pil = to_rgb_image(arr).resize((w, h))
        overlay_pil = cls._render_visual_overlay(base_pil, cleaned_mask, boxes, base_color)

        # Encode to Base64 PNG
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        overlay_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        mean_conf = round(float(np.mean([b["confidence"] for b in boxes])) if boxes else 0.45, 3)

        return {
            "target_concept": target_concept,
            "detected_feature": feature_name,
            "count": len(boxes),
            "coverage_percent": coverage_pct,
            "confidence": mean_conf,
            "boxes": boxes,
            "visual_evidence": {
                "type": "overlay_image",
                "image_base64": overlay_b64,
                "target": feature_name,
                "count": len(boxes),
                "coverage_percent": coverage_pct,
                "boxes": boxes,
            }
        }

    @classmethod
    def _fallback_find_boxes(
        cls,
        mask: np.ndarray,
        feature_name: str,
        h: int,
        w: int,
        max_boxes: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Pure NumPy bounding box extraction fallback.
        """
        boxes = []
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        if not np.any(rows) or not np.any(cols):
            return []

        # Find global bounding box and grid subdivisions if large
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        bw = max(1, xmax - xmin)
        bh = max(1, ymax - ymin)
        area = int(np.sum(mask > 0))

        boxes.append({
            "id": 1,
            "label": f"{feature_name}_1",
            "box_2d": [
                round(float(ymin) / h, 4),
                round(float(xmin) / w, 4),
                round(float(ymax) / h, 4),
                round(float(xmax) / w, 4),
            ],
            "pixel_coords": [int(ymin), int(xmin), int(ymax), int(xmax)],
            "confidence": 0.85,
            "area_pixels": area,
        })
        return boxes

    @classmethod
    def _generate_activation_mask(
        cls,
        arr: np.ndarray,
        modality: InputModality,
        target: str,
    ) -> Tuple[np.ndarray, str, Tuple[int, int, int]]:
        """
        Generate binary activation map based on image features and target concept.
        """
        c, h, w = arr.shape

        if any(k in target for k in ["water", "river", "lake", "ocean", "sea", "pond", "reservoir", "hydrological"]):
            feature_name = "water"
            color = cls.COLOR_MAP["water"]
            if c >= 4 and modality in (InputModality.MULTIMODAL_S1_S2, InputModality.SENTINEL2_MULTISPECTRAL):
                green = arr[1]
                nir = arr[3]
                denom = green + nir
                denom[denom == 0] = 1e-6
                ndwi = (green - nir) / denom
                mask = ndwi > 0.05
            else:
                r, g, b = arr[0], arr[1], arr[2]
                mask = (b >= r) & (g >= r * 0.9) & ((r + g + b) < 1.1) & (b > 0.08)

        elif any(k in target for k in ["tree", "forest", "vegetation", "canopy", "greenery", "woodland"]):
            feature_name = "vegetation"
            color = cls.COLOR_MAP["forest"]
            if c >= 4 and modality in (InputModality.MULTIMODAL_S1_S2, InputModality.SENTINEL2_MULTISPECTRAL):
                nir = arr[3]
                red = arr[2]
                denom = nir + red
                denom[denom == 0] = 1e-6
                ndvi = (nir - red) / denom
                mask = ndvi > 0.35
            else:
                r, g, b = arr[0], arr[1], arr[2]
                mask = (g > r * 1.05) & (g > b * 1.05) & (g > 0.15)

        elif any(k in target for k in ["building", "house", "urban", "roof", "structure", "built-up", "settlement"]):
            feature_name = "building"
            color = cls.COLOR_MAP["building"]
            if c >= 8 and modality in (InputModality.MULTIMODAL_S1_S2, InputModality.SENTINEL2_MULTISPECTRAL):
                swir = arr[7]
                nir = arr[3]
                denom = swir + nir
                denom[denom == 0] = 1e-6
                ndbi = (swir - nir) / denom
                mask = ndbi > 0.0
            else:
                r, g, b = arr[0], arr[1], arr[2]
                brightness = (r + g + b) / 3.0
                grey_diff = np.maximum(np.abs(r - g), np.abs(g - b))
                dy, dx = np.gradient(brightness)
                edges = np.sqrt(dx**2 + dy**2)
                mask = (grey_diff < 0.12) & (brightness > 0.25) & (brightness < 0.88) & (edges > 0.04)

        elif any(k in target for k in ["road", "highway", "street", "path", "runway", "corridor"]):
            feature_name = "road"
            color = cls.COLOR_MAP["road"]
            gray = np.mean(arr[:3], axis=0) if c >= 3 else arr[0]
            dy, dx = np.gradient(gray)
            edges = np.sqrt(dx**2 + dy**2)
            mask = (edges > 0.08) & (gray > 0.15) & (gray < 0.8)

        elif any(k in target for k in ["crop", "agriculture", "farmland", "field", "arable", "pasture"]):
            feature_name = "agriculture"
            color = cls.COLOR_MAP["agriculture"]
            if c >= 4 and modality in (InputModality.MULTIMODAL_S1_S2, InputModality.SENTINEL2_MULTISPECTRAL):
                nir = arr[3]
                red = arr[2]
                denom = nir + red
                denom[denom == 0] = 1e-6
                ndvi = (nir - red) / denom
                mask = (ndvi > 0.18) & (ndvi <= 0.45)
            else:
                r, g, b = arr[0], arr[1], arr[2]
                veg = (g > r) & (g > b)
                brightness = (r + g + b) / 3.0
                mask = (veg & (brightness > 0.35)) | ((r > 0.28) & (g > 0.25) & (b < 0.35))

        else:
            feature_name = target or "region"
            color = cls.COLOR_MAP["default"]
            gray = np.mean(arr[:3], axis=0) if c >= 3 else arr[0]
            thresh = np.percentile(gray, 75)
            mask = gray > thresh

        return mask, feature_name, color

    @classmethod
    def _render_visual_overlay(
        cls,
        base_img: Image.Image,
        mask: np.ndarray,
        boxes: List[Dict[str, Any]],
        color_rgb: Tuple[int, int, int],
    ) -> Image.Image:
        """
        Render semi-transparent highlight mask + crisp bounding boxes and labels onto base image.
        """
        w, h = base_img.size

        # 1. Translucent mask fill
        if _HAS_CV2:
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_pil = Image.fromarray(mask).resize((w, h), Image.NEAREST)
            mask_resized = np.array(mask_pil)

        color_rgba = (*color_rgb, 65)  # 25% opacity
        mask_indices = np.where(mask_resized > 0)
        mask_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        mask_overlay[mask_indices] = color_rgba
        mask_layer = Image.fromarray(mask_overlay, mode="RGBA")
        base_composite = Image.alpha_composite(base_img.convert("RGBA"), mask_layer)

        draw_comp = ImageDraw.Draw(base_composite)

        # 2. Draw crisp bounding boxes and badges
        box_color = (*color_rgb, 255)
        for b in boxes:
            y1, x1, y2, x2 = b["pixel_coords"]
            y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
            x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))

            draw_comp.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

            label_text = f"{b['label']} ({b['confidence']:.2f})"
            badge_h = 14
            badge_w = len(label_text) * 7 + 6
            badge_y1 = max(0, y1 - badge_h)
            badge_y2 = badge_y1 + badge_h
            badge_x2 = min(w, x1 + badge_w)

            draw_comp.rectangle([x1, badge_y1, badge_x2, badge_y2], fill=(20, 20, 20, 220))
            draw_comp.text((x1 + 3, badge_y1 + 1), label_text, fill=(255, 255, 255, 255))

        return base_composite.convert("RGB")
