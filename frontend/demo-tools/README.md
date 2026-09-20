# Demo Scene Generator (`make_scene.py`)

Offline utility script to prepare demo scenes for SatQuery's Change Detection page.

## Requirements
- Python 3.9+
- `Pillow`
- `numpy`

Install dependencies if missing:
```bash
pip install pillow numpy
```

## Usage
Place `before.jpeg` and `after.jpeg` (or `.jpg` / `.png`) into your scene directory under `public/demo/change/`, for example:
`public/demo/change/scene-01/`

Run the script from the `frontend/` directory:
```bash
python demo-tools/make_scene.py public/demo/change/scene-01
```

## How It Works
1. Resizes images to a common dimension (max 1024px on long side).
2. Computes mean absolute RGB difference with a 5x5 box blur filter.
3. Automatically evaluates the threshold (default: 50.0) to ensure only visibly changed regions are marked (threshold increases if change > 35%).
4. Classifies changed pixels into "Vegetation loss", "Built-up increase", "Water and shoreline change", and "Other change".
5. Calculates spatial centroids (`upper-left`, `upper-right`, `lower-left`, `lower-right`, `center`) for categories with >1% change.
6. Outputs:
   - `diff.png`: Multi-channel change heatmap at full native resolution.
   - `result.json`: Exact computed metrics, spatial positions, and category breakdown.
   - `manifest.json`: Scene registration in `public/demo/change/manifest.json`.
