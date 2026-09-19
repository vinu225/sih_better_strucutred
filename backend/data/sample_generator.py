from pathlib import Path
import numpy as np
import torch
from backend.config import SAMPLES_DIR, IMAGE_SIZE
from backend.data.dummy_dataset import create_synthetic_signature
from backend.data.sentinel_loader import render_rgb_composite


def generate_sample_tiles(samples_dir: Path = SAMPLES_DIR):
    """
    Generate sample 12-channel multispectral numpy tiles and preview RGB images.
    """
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_types = {
        "sample_forest_tile": "forest",
        "sample_water_tile": "water",
        "sample_urban_tile": "urban",
        "sample_cropland_tile": "cropland",
    }

    generated_paths = {}
    for name, terrain in sample_types.items():
        npy_path = samples_dir / f"{name}.npy"
        png_path = samples_dir / f"{name}_rgb.png"

        # Generate (12, 120, 120) tensor
        tensor = create_synthetic_signature(terrain, height=IMAGE_SIZE, width=IMAGE_SIZE)
        np.save(str(npy_path), tensor.numpy())

        # Save RGB quick-look
        rgb_img = render_rgb_composite(tensor)
        rgb_img.save(str(png_path))

        generated_paths[name] = {"npy": str(npy_path), "png": str(png_path), "terrain": terrain}

    return generated_paths


if __name__ == "__main__":
    paths = generate_sample_tiles()
    print(f"Generated {len(paths)} sample tiles in {SAMPLES_DIR}")
