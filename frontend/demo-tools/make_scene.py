#!/usr/bin/env python3
"""
make_scene.py - Offline helper script to pre-process a Change Detection scene from before.jpeg and after.jpeg.
Computes per-pixel change magnitude between before.jpeg and after.jpeg,
generates a difference heatmap (diff.png), computes pixel statistics,
spatial centroids of change categories, and generates result.json + manifest.json.

Dependencies: Pillow, numpy (standard Python packages).
Usage:
    python demo-tools/make_scene.py public/demo/change/scene-01
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageFilter

# Configurable constants
MAX_IMAGE_DIM = 1024
BOX_BLUR_RADIUS = 2     # 5x5 box blur (radius 2)
INITIAL_THRESHOLD = 50.0

def get_centroid_quadrant(mask):
    """
    Computes mask centroid and translates it into one of:
    'upper-left', 'upper-right', 'lower-left', 'lower-right', or 'center'.
    """
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return "center"

    h, w = mask.shape
    mean_y = np.mean(y_indices) / h
    mean_x = np.mean(x_indices) / w

    # Define center zone between 0.35 and 0.65
    is_center_x = 0.35 <= mean_x <= 0.65
    is_center_y = 0.35 <= mean_y <= 0.65

    if is_center_x and is_center_y:
        return "center"
    
    vert = "upper" if mean_y < 0.5 else "lower"
    horiz = "left" if mean_x < 0.5 else "right"
    return f"{vert}-{horiz}"

def find_image_files(scene_dir):
    # Primary: before.jpeg and after.jpeg
    for ext in [".jpeg", ".jpg", ".png"]:
        b = os.path.join(scene_dir, f"before{ext}")
        a = os.path.join(scene_dir, f"after{ext}")
        if os.path.exists(b) and os.path.exists(a):
            return b, a
    return None, None

def compute_scene(scene_dir, threshold=INITIAL_THRESHOLD):
    before_path, after_path = find_image_files(scene_dir)
    if not before_path or not after_path:
        print(f"Error: Could not find before.jpeg and after.jpeg in {scene_dir}")
        sys.exit(1)

    print(f"Loading before image: {before_path}")
    print(f"Loading after image:  {after_path}")

    img_before_orig = Image.open(before_path).convert("RGB")
    img_after_orig = Image.open(after_path).convert("RGB")

    orig_w, orig_h = img_before_orig.size

    # Resize to common size (max 1024 on long side)
    scale = min(1.0, MAX_IMAGE_DIM / max(orig_w, orig_h))
    proc_w = int(orig_w * scale)
    proc_h = int(orig_h * scale)

    img_before = img_before_orig.resize((proc_w, proc_h), Image.Resampling.LANCZOS)
    img_after = img_after_orig.resize((proc_w, proc_h), Image.Resampling.LANCZOS)

    arr_before = np.array(img_before, dtype=np.float32)
    arr_after = np.array(img_after, dtype=np.float32)

    # 1. Mean absolute RGB difference
    abs_diff = np.mean(np.abs(arr_before - arr_after), axis=2)

    # 2. 5x5 Box Blur (radius 2)
    diff_pil = Image.fromarray(np.clip(abs_diff, 0, 255).astype(np.uint8))
    blurred_diff_pil = diff_pil.filter(ImageFilter.BoxBlur(BOX_BLUR_RADIUS))
    smoothed_diff = np.array(blurred_diff_pil, dtype=np.float32)

    # Check threshold and adjust if changed_percent > 35%
    curr_threshold = threshold
    changed_mask = smoothed_diff >= curr_threshold
    changed_percent = (float(np.sum(changed_mask)) / changed_mask.size) * 100.0

    while changed_percent > 35.0 and curr_threshold < 150.0:
        curr_threshold += 5.0
        changed_mask = smoothed_diff >= curr_threshold
        changed_percent = (float(np.sum(changed_mask)) / changed_mask.size) * 100.0

    print(f"\nFinal Parameters:")
    print(f"  Max Dimension: {MAX_IMAGE_DIM} (Processed Size: {proc_w}x{proc_h})")
    print(f"  Blur: 5x5 box blur (radius {BOX_BLUR_RADIUS})")
    print(f"  Threshold: {curr_threshold:.1f}")
    print(f"  Changed Pixels Percentage: {changed_percent:.2f}%")

    total_pixels = changed_mask.size
    changed_pct_rounded = round(changed_percent, 1)

    # 3. Classify categories
    # Greenness index: G - (R + B)/2
    greenness_before = arr_before[:, :, 1] - 0.5 * (arr_before[:, :, 0] + arr_before[:, :, 2])
    greenness_after = arr_after[:, :, 1] - 0.5 * (arr_after[:, :, 0] + arr_after[:, :, 2])

    # Water proxy: blue-dominant or low brightness
    water_before = (arr_before[:, :, 2] > arr_before[:, :, 0] + 5.0) & (arr_before[:, :, 2] > arr_before[:, :, 1]) | (np.mean(arr_before, axis=2) < 40.0)

    # Brightness / Urban proxy: mean intensity increase
    brightness_before = np.mean(arr_before, axis=2)
    brightness_after = np.mean(arr_after, axis=2)

    # Category masks
    veg_loss_mask = changed_mask & (greenness_before > 10.0) & (greenness_after < greenness_before - 8.0)
    water_change_mask = changed_mask & (~veg_loss_mask) & water_before
    urban_inc_mask = changed_mask & (~veg_loss_mask) & (~water_change_mask) & (brightness_after > brightness_before + 12.0)
    other_mask = changed_mask & (~veg_loss_mask) & (~water_change_mask) & (~urban_inc_mask)

    veg_loss_pct = round(float((np.sum(veg_loss_mask) / total_pixels) * 100), 1)
    urban_inc_pct = round(float((np.sum(urban_inc_mask) / total_pixels) * 100), 1)
    water_change_pct = round(float((np.sum(water_change_mask) / total_pixels) * 100), 1)
    other_pct = round(float(changed_pct_rounded - veg_loss_pct - urban_inc_pct - water_change_pct), 1)
    if other_pct < 0:
        other_pct = 0.0

    # Centroids for categories > 1%
    veg_pos = get_centroid_quadrant(veg_loss_mask) if veg_loss_pct >= 1.0 else "center"
    urban_pos = get_centroid_quadrant(urban_inc_mask) if urban_inc_pct >= 1.0 else "center"
    water_pos = get_centroid_quadrant(water_change_mask) if water_change_pct >= 1.0 else "center"

    def pos_phrase(pos):
        if pos == "center":
            return "in the central part of the image"
        return f"mostly in the {pos} part of the image"

    categories = [
        {"label": "Vegetation loss", "percent": veg_loss_pct},
        {"label": "Built-up increase", "percent": urban_inc_pct},
        {"label": "Water and shoreline change", "percent": water_change_pct},
        {"label": "Other change", "percent": other_pct}
    ]

    # 4. Generate diff.png RGBA at full original before.jpeg dimensions
    # Upsample smoothed_diff back to original size
    diff_full_pil = Image.fromarray(np.clip(smoothed_diff, 0, 255).astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    smoothed_full = np.array(diff_full_pil, dtype=np.float32)
    changed_full_mask = smoothed_full >= curr_threshold

    diff_rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    max_val = max(float(np.max(smoothed_full)), curr_threshold + 25.0)
    norm_val = np.clip((smoothed_full - curr_threshold) / (max_val - curr_threshold + 1e-5), 0.0, 1.0)

    # Yellow (255, 235, 59) to Orange (255, 140, 0) to Red (239, 68, 68)
    diff_rgba[:, :, 0] = 255
    diff_rgba[:, :, 1] = np.clip(235.0 * (1.0 - norm_val * 0.7), 0, 255).astype(np.uint8)
    diff_rgba[:, :, 2] = np.clip(59.0 * (1.0 - norm_val), 0, 255).astype(np.uint8)
    diff_rgba[:, :, 3] = (changed_full_mask.astype(np.float32) * (150.0 + norm_val * 90.0)).astype(np.uint8)

    diff_path = os.path.join(scene_dir, "diff.png")
    Image.fromarray(diff_rgba, mode="RGBA").save(diff_path, format="PNG")
    print(f"Saved diff heatmap: {diff_path} ({orig_w}x{orig_h})")

    # 5. Build strict answers using only derived numbers and quadrant positions
    answers = {
        "overview": (
            f"Comparing the two scenes reveals an estimated {changed_pct_rounded}% overall surface area modification. "
            f"The primary changes include {veg_loss_pct}% vegetation loss {pos_phrase(veg_pos)} and {urban_inc_pct}% built-up expansion {pos_phrase(urban_pos)}. "
            f"Use the divider slider above to compare the two time periods directly."
        ),
        "vegetation": (
            f"Vegetation analysis indicates an estimated {veg_loss_pct}% reduction in dense green canopy, concentrated {pos_phrase(veg_pos)}. "
            f"Sliding the divider reveals where green cover has altered across the landscape. You can ask about built-up expansion or water bodies for further details."
        ),
        "deforestation": (
            f"Canopy coverage data shows an estimated {veg_loss_pct}% loss in tree and forest cover, situated {pos_phrase(veg_pos)}. "
            f"The comparison highlights cleared patches where original dense canopy has transitioned to open ground. You can use the slider to contrast earlier tree margins."
        ),
        "urban": (
            f"Built-up terrain and impervious surfaces increased by an estimated {urban_inc_pct}%, concentrated {pos_phrase(urban_pos)}. "
            f"The later observation highlights new structures, roads, and developed parcels. Drag the slider to inspect the exact progression of new infrastructure."
        ),
        "water": (
            f"Hydrological and shoreline features show an estimated {water_change_pct}% variation {pos_phrase(water_pos)}. "
            f"Surface water boundaries and moisture margins exhibit localized adjustments between the two time periods. Sliding the divider allows you to inspect shoreline changes."
        ),
        "howMuch": (
            f"Total surface change across the evaluated scene is estimated at {changed_pct_rounded}%. "
            f"This comprises {veg_loss_pct}% vegetation loss, {urban_inc_pct}% built-up increase, {water_change_pct}% water change, and {other_pct}% other surface modifications. "
            f"The detailed analysis card below displays the complete category breakdown."
        ),
        "showDifference": (
            f"The difference map highlights areas exceeding the change detection threshold, covering {changed_pct_rounded}% of the total scene. "
            f"High-magnitude modifications appear in yellow to red highlights across the image. You can adjust the difference overlay opacity using the slider controls above."
        ),
        "fallback": (
            f"Temporal comparison shows noticeable landscape modifications across {changed_pct_rounded}% of the image. "
            f"Moving the slider divider allows you to inspect specific features side by side. Ask about vegetation, built-up areas, or water bodies for category details."
        )
    }

    result_data = {
        "before_label": None,
        "after_label": None,
        "location": None,
        "illustrative": True,
        "stats": {
            "changed_percent": changed_pct_rounded,
            "increase_percent": urban_inc_pct,
            "decrease_percent": veg_loss_pct,
            "categories": categories
        },
        "tools_used": [
            "Change detection"
        ],
        "answers": answers,
        "follow_ups": [
            "What changed in vegetation?",
            "Show urban expansion",
            "How much total area changed?"
        ]
    }

    result_path = os.path.join(scene_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
    print(f"Saved result.json: {result_path}")

    # Update manifest.json
    parent_dir = os.path.dirname(os.path.abspath(scene_dir))
    manifest_path = os.path.join(parent_dir, "manifest.json")
    scene_id = os.path.basename(os.path.abspath(scene_dir))

    manifest_data = [
        {
            "id": scene_id,
            "title": "Temporal Satellite Comparison",
            "illustrative": True
        }
    ]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved manifest.json: {manifest_path}")

    print("\n================ FINAL COMPUTED METRICS ================")
    print(f"Total Changed:              {changed_pct_rounded}%")
    print(f"Vegetation Loss:            {veg_loss_pct}% ({veg_pos})")
    print(f"Built-up Increase:          {urban_inc_pct}% ({urban_pos})")
    print(f"Water and Shoreline Change: {water_change_pct}% ({water_pos})")
    print(f"Other Change:               {other_pct}%")
    print("========================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Pre-process change detection sample scene from before.jpeg and after.jpeg.")
    parser.add_argument("scene_dir", help="Path to scene directory (e.g. public/demo/change/scene-01)")
    parser.add_argument("--threshold", type=float, default=INITIAL_THRESHOLD, help="Initial change threshold")
    args = parser.parse_args()
    compute_scene(args.scene_dir, args.threshold)

if __name__ == "__main__":
    main()
