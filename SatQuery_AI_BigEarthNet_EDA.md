# BigEarthNet.txt — Complete EDA Plan & Framework for SatQuery AI (SIH26167)

## 0. Access & Methodology Statement (read this first)

- I do **not** have live code-execution access to `huggingface.co` from this environment, and `web_fetch` only returns the rendered HTML preview (100 rows), not the raw Parquet file.
- Every number below is one of exactly three things, and each is labeled:
  - **[CONFIRMED]** — stated on the official dataset card / arXiv paper.
  - **[COMPUTED]** — arithmetic derived directly from CONFIRMED numbers (shown, not hidden).
  - **[SAMPLE, n=100]** — observed in the 100-row HF preview only. Never representative of 9.55M rows — used only to demonstrate real structure/examples, never for imbalance or leakage conclusions.
- Everything else (missing values, duplicates, split leakage, box validity, text-length percentiles, country/season distributions, actual imbalance ratios) is **unknown until you run the code below**. I will not guess these.

**Run this first** — it gets you every number this report is missing, in one pass, without downloading 467MB into memory:

```python
# pip install duckdb
import duckdb

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

PARQUET = "https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/resolve/main/BigEarthNet.txt.parquet"
# Alternative if your duckdb build supports the hf:// protocol (duckdb >= 0.10.1):
# PARQUET = "hf://datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/BigEarthNet.txt.parquet"

con.execute(f"CREATE VIEW ben AS SELECT * FROM read_parquet('{PARQUET}')")
print(con.execute("SELECT COUNT(*) AS total_rows FROM ben").df())
```

If `httpfs`/remote read is unavailable in your network, `git clone` the repo (the dataset card's own instructions) and point `read_parquet('BigEarthNet.txt.parquet')` at the local file instead — everything below works identically either way.

---

## 1. Dataset Overview

**[CONFIRMED]**
- Total annotation rows: 9,553,962
- Unique image pairs (S1+S2): 464,044
- Benchmark split: 1,082 image pairs / 15,029 annotations
- File size: 467 MB (Parquet)
- Columns (13): `ID`, `s1_name`, `patch_id`, `input`, `output`, `type`, `category`, `split`, `latitude`, `longitude`, `country`, `season`, `climate_zone`
- `type`: 4 distinct values (binary, mcq, captioning, bounding box)
- `category`: 11 distinct values (10 observed in sample: adjacency, area, count, presence, point, reference, climate zone, country, season, relative pos)
- `split`: 4 distinct values (train, validation, test, bench)

**[COMPUTED]**
- Average annotations per image pair = 9,553,962 / 464,044 ≈ **20.59**
- Average annotations per bench pair = 15,029 / 1,082 ≈ **13.89** (lower — consistent with "manually verified" being a curated, not auto-generated, subset)

**Column meanings** (from dataset card, already covered in the previous turn — not repeating in full here; see `ID`=row key, `s1_name`/`patch_id`=SAR/optical patch identifiers, `input`/`output`=question/answer, `type`/`category`=task taxonomy, `split`=ML split, `latitude`/`longitude`/`country`/`season`/`climate_zone`=geo-metadata).

**Code — full overview:**
```python
print(con.execute("""
    SELECT
        COUNT(*)                       AS total_rows,
        COUNT(DISTINCT ID)             AS unique_ids,
        COUNT(DISTINCT patch_id)       AS unique_s2_patches,
        COUNT(DISTINCT s1_name)        AS unique_s1_images,
        COUNT(DISTINCT (s1_name, patch_id)) AS unique_pairs,
        COUNT(DISTINCT type)           AS n_types,
        COUNT(DISTINCT category)       AS n_categories,
        COUNT(DISTINCT split)          AS n_splits,
        COUNT(DISTINCT country)        AS n_countries,
        COUNT(DISTINCT season)         AS n_seasons,
        COUNT(DISTINCT climate_zone)   AS n_climate_zones
    FROM ben
""").df())

print(con.execute("DESCRIBE ben").df())  # dtypes
```

---

## 2. Data Quality

**Nothing here is known yet — this entire section is "INVESTIGATE" until the code runs.** Run this, then classify each result KEEP / FIX / REMOVE / NORMALIZE / INVESTIGATE per the rule of thumb noted after the code.

```python
report = {}

# Nulls per column
report["nulls"] = con.execute("""
    SELECT
      sum(CASE WHEN ID IS NULL THEN 1 ELSE 0 END) AS null_id,
      sum(CASE WHEN s1_name IS NULL THEN 1 ELSE 0 END) AS null_s1,
      sum(CASE WHEN patch_id IS NULL THEN 1 ELSE 0 END) AS null_patch,
      sum(CASE WHEN input IS NULL OR trim(input)='' THEN 1 ELSE 0 END) AS null_input,
      sum(CASE WHEN output IS NULL OR trim(output)='' THEN 1 ELSE 0 END) AS null_output,
      sum(CASE WHEN latitude IS NULL THEN 1 ELSE 0 END) AS null_lat,
      sum(CASE WHEN longitude IS NULL THEN 1 ELSE 0 END) AS null_lon
    FROM ben
""").df()

# Full duplicate rows
report["dup_rows"] = con.execute("""
    SELECT COUNT(*) - COUNT(DISTINCT (s1_name,patch_id,input,output,type,category)) AS dup_count
    FROM ben
""").df()

# Duplicate input-output pairs (per image, could be legitimate near-duplicates from templating)
report["dup_qa_per_image"] = con.execute("""
    SELECT patch_id, input, output, COUNT(*) AS n
    FROM ben GROUP BY patch_id, input, output HAVING COUNT(*) > 1
    ORDER BY n DESC LIMIT 20
""").df()

# Invalid coordinates
report["bad_coords"] = con.execute("""
    SELECT COUNT(*) FROM ben
    WHERE latitude < -90 OR latitude > 90 OR longitude < -180 OR longitude > 180
""").df()

# Inconsistent categorical casing/whitespace (a common silent bug)
report["type_values"] = con.execute("SELECT DISTINCT type FROM ben").df()
report["category_values"] = con.execute("SELECT DISTINCT category FROM ben").df()
report["split_values"] = con.execute("SELECT DISTINCT split FROM ben").df()
report["country_values"] = con.execute("SELECT DISTINCT country FROM ben ORDER BY 1").df()
report["season_values"] = con.execute("SELECT DISTINCT season FROM ben ORDER BY 1").df()

for k, v in report.items():
    print(f"\n--- {k} ---")
    print(v)
```

**Classification rule of thumb to apply to your results** (apply this logic once you have numbers — don't apply it blind):
- Null `input`/`output` → **REMOVE** (unusable for training, both text-supervised objectives need both fields).
- Null lat/long on a small fraction → **INVESTIGATE** (could still train on text/task alone; matters for geo-eval only).
- Exact duplicate rows → **REMOVE** (pure redundancy, risks weighting some pairs' phrasing more than others).
- Repeated `input`/`output` *per image* with different `ID` → likely template artifacts (expected, since captions/MCQs are template-generated per the paper) → **KEEP but flag** — this affects section 11/15, not necessarily a defect.
- Casing/whitespace inconsistency in `category`/`country`/`season` → **NORMALIZE** (lowercase/strip before any groupby, or every downstream stat you compute is silently wrong).
- Out-of-range lat/long → **FIX or REMOVE** depending on count (single-digit issues: drop; systemic: investigate the source pipeline).

---

## 3. Split Analysis — Leakage Check (critical)

```python
print(con.execute("""
    SELECT split,
           COUNT(*) AS annotations,
           COUNT(DISTINCT (s1_name,patch_id)) AS unique_pairs,
           ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2) AS pct_annotations
    FROM ben GROUP BY split ORDER BY annotations DESC
""").df())

# THE critical leakage check: does the same image pair appear in more than one split?
leakage = con.execute("""
    WITH pair_splits AS (
        SELECT s1_name, patch_id, COUNT(DISTINCT split) AS n_splits,
               STRING_AGG(DISTINCT split, ',') AS splits_seen
        FROM ben GROUP BY s1_name, patch_id
    )
    SELECT * FROM pair_splits WHERE n_splits > 1
""").df()
print(f"\nImage pairs appearing in >1 split: {len(leakage)}")
print(leakage.head(20))
```

**Why annotation-level random splitting is dangerous here (this is a real, structural risk — not conditional on running the code):**
Because one image pair generates ~20.6 annotations on average [COMPUTED above], a naive `train_test_split` on the 9.55M *rows* will almost certainly place different questions about the *same* image into both train and test. The model then doesn't have to generalize to a new image at test time — it can partially memorize the specific patch's visual content from training and just answer a *different question* about it. Your test accuracy would be inflated and would not reflect real-world performance on genuinely unseen satellite scenes. This is the single most likely evaluation bug for a hackathon team under time pressure.

**Recommended split strategy:** split by unique `(s1_name, patch_id)` pair first (GroupKFold / group-aware split, grouped on `patch_id`), then take *all* annotations for a pair into whichever split that pair was assigned to. If the dataset's own `split` column already does this correctly (plausible, since it's a purpose-built research dataset), your leakage query above should return 0 — confirm it before trusting the provided splits, don't assume it.

---

## 4. Task Distribution (`type`)

```python
print(con.execute("""
    SELECT type,
           COUNT(*) AS n,
           ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2) AS pct,
           COUNT(DISTINCT patch_id) AS unique_images,
           ROUND(COUNT(*)*1.0/COUNT(DISTINCT patch_id), 2) AS annotations_per_image
    FROM ben GROUP BY type ORDER BY n DESC
""").df())
```

**ML/DL meaning per type:**
- `binary` → classification head, sigmoid/2-class, needs balance check (Section 7).
- `mcq` → classification over a variable option set — options are embedded in `input` text, not a fixed label space, so this needs prompt-conditioned classification or generative answer matching, not a plain softmax head.
- `captioning` → sequence generation, needs a language-model decoder and generation metrics (BLEU/ROUGE/BERTScore).
- `bounding box` → regression/localization output, needs IoU-based evaluation, and your model architecture needs an explicit coordinate-output pathway (not just free-text generation) or you need to parse coordinates out of generated text reliably.

This is exactly why a single-head classifier cannot answer this dataset — you need either a **task-router architecture** (detect task type from the query, dispatch to the right output head) or a **fully generative VLM** that emits task-appropriate text (a raw coordinate string for bbox, a letter for mcq, free text for captioning) and you parse/validate the output post-hoc.

---

## 5. Category Distribution

```python
print(con.execute("""
    SELECT type, category,
           COUNT(*) AS n,
           ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2) AS pct,
           COUNT(DISTINCT patch_id) AS unique_images
    FROM ben GROUP BY type, category ORDER BY n DESC
""").df())

# Type x Category heatmap data
heat = con.execute("""
    SELECT type, category, COUNT(*) AS n FROM ben GROUP BY type, category
""").df()
pivot = heat.pivot(index="category", columns="type", values="n").fillna(0)
print(pivot)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis")
plt.title("Type x Category Annotation Counts")
plt.xlabel("Type"); plt.ylabel("Category")
plt.tight_layout(); plt.savefig("type_category_heatmap.png")
```

Do not assume which categories are "dominant" or "rare" until this returns real counts — with 11 categories across 4 types and templated generation, imbalance is plausible but the *direction and severity* is unknown to me right now.

---

## 6. Text / NLP EDA (`input`)

```python
print(con.execute("""
    SELECT
        MIN(length(input)) AS min_len,
        MAX(length(input)) AS max_len,
        AVG(length(input)) AS mean_len,
        MEDIAN(length(input)) AS median_len,
        STDDEV(length(input)) AS std_len,
        QUANTILE_CONT(length(input), 0.25) AS q1,
        QUANTILE_CONT(length(input), 0.75) AS q3,
        QUANTILE_CONT(length(input), 0.90) AS p90,
        QUANTILE_CONT(length(input), 0.95) AS p95
    FROM ben
""").df())

# by task/category
print(con.execute("""
    SELECT type, category,
        AVG(length(input)) AS mean_char_len,
        AVG(length(input) - length(replace(input,' ',''))+1) AS approx_mean_word_count
    FROM ben GROUP BY type, category ORDER BY mean_char_len DESC
""").df())

# Repeated exact questions (template detection)
print(con.execute("""
    SELECT input, COUNT(*) AS n FROM ben GROUP BY input ORDER BY n DESC LIMIT 30
""").df())

# Special tokens
print(con.execute("""
    SELECT
      sum(CASE WHEN input LIKE '%<point>%' THEN 1 ELSE 0 END) AS has_point_token,
      sum(CASE WHEN input LIKE '%<ref>%' THEN 1 ELSE 0 END) AS has_ref_token
    FROM ben
""").df())
```

**[SAMPLE, n=100] — confirmed real patterns already visible** (not full-dataset stats, just proof these phenomena exist): the preview shows heavy template reuse — e.g. "Provide a bounding box for the land cover class instance at `<point>(x,y)</point>` in the satellite image." recurs with only coordinates changed; `<point>`/`<ref>` are real literal tokens embedded in `input` text, not metadata fields — your tokenizer must treat them as structural markers (special tokens), not natural words, or the model will not learn their grounding function.

**Implication:** heavy templating means high surface-level n-gram redundancy. This is good for a VLM to learn task-format compliance quickly, but bad if you evaluate on `test`/`bench` without checking whether the *same templates* (just different images) leaked in from `train` — this is a second, subtler leakage vector beyond image-level leakage (Section 3): template memorization instead of true visual grounding.

---

## 7. Output / Answer EDA

```python
# Binary balance
print(con.execute("""
    SELECT output, COUNT(*) AS n FROM ben WHERE type='binary' GROUP BY output
""").df())

# MCQ option letter balance (answer key distribution)
print(con.execute("""
    SELECT output, COUNT(*) AS n FROM ben WHERE type='mcq' GROUP BY output ORDER BY n DESC
""").df())

# Captioning length
print(con.execute("""
    SELECT AVG(length(output)) AS mean_len, MEDIAN(length(output)) AS median_len,
           MIN(length(output)) AS min_len, MAX(length(output)) AS max_len
    FROM ben WHERE type='captioning'
""").df())

# Bounding box parsing + validity — format observed as "[x1 y1, x2 y2]" in normalized 0-1 coords
import re
bbox_df = con.execute("SELECT ID, output FROM ben WHERE type='bounding box'").df()

def parse_bbox(s):
    m = re.findall(r"[-+]?\d*\.?\d+", s)
    return [float(x) for x in m] if len(m) == 4 else None

bbox_df["coords"] = bbox_df["output"].apply(parse_bbox)
bbox_df = bbox_df.dropna(subset=["coords"])
bbox_df[["x1","y1","x2","y2"]] = bbox_df["coords"].apply(pd.Series)

bbox_df["valid_order"] = (bbox_df.x1 < bbox_df.x2) & (bbox_df.y1 < bbox_df.y2)
bbox_df["area"] = (bbox_df.x2 - bbox_df.x1) * (bbox_df.y2 - bbox_df.y1)

print("Invalid (x1>=x2 or y1>=y2):", (~bbox_df.valid_order).sum())
print(bbox_df["area"].describe())
print("Tiny boxes (<1% area):", (bbox_df.area < 0.01).sum())
print("Huge boxes (>80% area):", (bbox_df.area > 0.80).sum())
```

**MCQ note (structural, not sample-dependent):** the answer key (`output`) is a letter (a/b/c/d), but the *meaning* of each letter changes per question, since options are shuffled into the `input` text. This means you cannot treat `output` as a fixed-vocabulary classification target across the dataset — the model must condition on the specific `input` to know what "b" refers to. A naive "predict the most common letter" baseline is a real risk to check for (MCQ letter-position bias is a well-known dataset artifact in VQA literature) — run the balance query above before trusting any MCQ accuracy number.

---

## 8–9. Geographic + Season/Climate EDA

```python
print(con.execute("SELECT country, COUNT(*) n, COUNT(DISTINCT patch_id) imgs FROM ben GROUP BY country ORDER BY n DESC").df())
print(con.execute("SELECT season, COUNT(*) n FROM ben GROUP BY season ORDER BY n DESC").df())
print(con.execute("SELECT climate_zone, COUNT(*) n FROM ben GROUP BY climate_zone ORDER BY n DESC").df())
print(con.execute("SELECT country, season, COUNT(*) n FROM ben GROUP BY country, season ORDER BY country, season").df())
print(con.execute("SELECT country, type, COUNT(*) n FROM ben GROUP BY country, type ORDER BY country, type").df())

# geographic scatter
geo = con.execute("SELECT DISTINCT patch_id, latitude, longitude, country FROM ben").df()
plt.figure(figsize=(8,6))
sns.scatterplot(data=geo, x="longitude", y="latitude", hue="country", s=5, legend="brief")
plt.title("Geographic distribution of unique image pairs")
plt.tight_layout(); plt.savefig("geo_scatter.png")
```

**Known structural constraint (not from this EDA — from BigEarthNet's own documented construction):** the underlying imagery covers 10 European countries only, acquired 2017–2018. **This matters directly for your ISRO/India use case:** whatever you fine-tune here will have zero exposure to Indian terrain, crop patterns, monsoon-season imagery, or Indian administrative/LULC class distributions. This is a domain-shift risk you should state explicitly to your team and judges — it's not a data quality bug, it's a scope mismatch between the fine-tuning dataset and the deployment domain, and it should shape your evaluation story (e.g., testing generalization qualitatively on any Indian Sentinel scene you can get, even without labels).

---

## 10. Annotation Density — "9.55M annotations does NOT mean 9.55M independent images"

**[COMPUTED, already explained in Section 1]:** 9,553,962 annotations ÷ 464,044 unique image pairs ≈ **20.6 annotations per image on average**.

**Why this must be said explicitly to your team:** if someone reports "we have 9.55 million training examples," that phrasing implies 9.55 million *independent visual observations* — it does not. You have **464,044 independent visual observations**, each asked about roughly 20 different ways. For any *vision*-side generalization claim (does the model actually understand SAR/optical imagery, versus does it understand English question templates), the meaningful denominator is 464K, not 9.55M. This directly feeds back into Section 3 (split-by-image, not split-by-row) and Section 6 (template memorization risk).

```python
density = con.execute("""
    SELECT patch_id, COUNT(*) AS n_annotations
    FROM ben GROUP BY patch_id
""").df()
print(density["n_annotations"].describe())
print("Extreme cases (top 10 most-annotated images):")
print(density.sort_values("n_annotations", ascending=False).head(10))
```

---

## 11. Duplication / Template Analysis

```python
print(con.execute("SELECT COUNT(*) - COUNT(DISTINCT input) AS exact_dup_questions FROM ben").df())

# normalized (lowercase, strip punctuation) duplicate check
import re as _re
sample = con.execute("SELECT input FROM ben USING SAMPLE 200000").df()  # sample for speed on 9.5M rows
sample["norm"] = sample["input"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()
print("Normalized duplicate rate (on 200k sample):", 1 - sample["norm"].nunique()/len(sample))
```

**Answer to "could the model memorize patterns":** yes, plausibly — given heavy templating (Section 6) and ~20.6 questions per image (Section 10), a model can learn to pattern-match "Do pastures cover between X and Y square meters?" → answer heuristics from phrasing alone, without truly grounding in pixels, especially for `binary`/`mcq` types. This is a real risk specific to this dataset's generation method (the paper describes template-filling + self-refinement for caption generation, which is efficient but structurally repetitive) and should shape your evaluation (favor `bench` split results over `test`, and spot-check qualitatively on held-out imagery your team hasn't seen in any split).

---

## 12. Representative Examples — [SAMPLE, n=100, genuinely real rows]

From the actual preview rows I retrieved earlier (Austria, summer, patch `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57` and neighbors):

- **Binary:** Q: "Would you say that any arable land lies next to pastures in the image?" → A: "yes"
- **MCQ:** Q: "Which classes share a boundary? a) Broad-leaved forest and Pastures, b) Coastal wetlands and Coniferous forest, c) Coniferous forest and Mixed forest, d) Arable land and Pastures" → A: "d"
- **Captioning:** "This satellite image, captured in Austria during summer, depicts a diverse landscape dominated by agricultural and forested areas..." (long-form, geographically anchored)
- **Bounding box (point→box):** Q: "Provide a bounding box for the land cover class instance at `<point>(0.82, 0.28)</point>` in the satellite image." → A: "[0.64 0.0, 1.0 0.71]"
- **Bounding box (referring expression):** Q: "Identify the location of the `<ref>largest connected region of pastures</ref>`." → A: "[0.0 0.33, 0.28 0.8]"

"Easy" vs "difficult" examples can't be honestly labeled from 100 rows — that judgment needs model-in-the-loop error analysis after a baseline exists, not pre-hoc guessing.

---

## 13. Visualizations — code for all 16 requested plots

```python
import matplotlib.pyplot as plt
import seaborn as sns

def barplot(df, x, y, title, xlabel, ylabel, fname, rotate=45):
    plt.figure(figsize=(9,5))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout(); plt.savefig(fname); plt.close()

# 1. split distribution
d = con.execute("SELECT split, COUNT(*) n FROM ben GROUP BY split").df()
barplot(d, "split", "n", "Annotation count by split", "Split", "Annotations", "01_split_dist.png")

# 2. task (type) distribution
d = con.execute("SELECT type, COUNT(*) n FROM ben GROUP BY type").df()
barplot(d, "type", "n", "Annotation count by task type", "Task type", "Annotations", "02_task_dist.png")

# 3. category distribution
d = con.execute("SELECT category, COUNT(*) n FROM ben GROUP BY category ORDER BY n DESC").df()
barplot(d, "category", "n", "Annotation count by category", "Category", "Annotations", "03_category_dist.png")

# 4. country distribution
d = con.execute("SELECT country, COUNT(*) n FROM ben GROUP BY country ORDER BY n DESC").df()
barplot(d, "country", "n", "Annotation count by country", "Country", "Annotations", "04_country_dist.png")

# 5. season distribution
d = con.execute("SELECT season, COUNT(*) n FROM ben GROUP BY season").df()
barplot(d, "season", "n", "Annotation count by season", "Season", "Annotations", "05_season_dist.png")

# 6. climate distribution
d = con.execute("SELECT climate_zone, COUNT(*) n FROM ben GROUP BY climate_zone ORDER BY n DESC").df()
barplot(d, "climate_zone", "n", "Annotation count by climate zone", "Climate zone", "Annotations", "06_climate_dist.png")

# 7. input length histogram
d = con.execute("SELECT length(input) AS l FROM ben USING SAMPLE 300000").df()
plt.figure(figsize=(8,5)); sns.histplot(d["l"], bins=50)
plt.title("Input (question) length distribution — 300k sample"); plt.xlabel("Characters"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig("07_input_len_hist.png"); plt.close()

# 8. output length by type (captioning)
d = con.execute("SELECT length(output) AS l FROM ben WHERE type='captioning'").df()
plt.figure(figsize=(8,5)); sns.histplot(d["l"], bins=50)
plt.title("Caption length distribution"); plt.xlabel("Characters"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig("08_caption_len_hist.png"); plt.close()

# 9. binary balance
d = con.execute("SELECT output, COUNT(*) n FROM ben WHERE type='binary' GROUP BY output").df()
barplot(d, "output", "n", "Binary answer balance", "Answer", "Count", "09_binary_balance.png", rotate=0)

# 10. mcq balance
d = con.execute("SELECT output, COUNT(*) n FROM ben WHERE type='mcq' GROUP BY output ORDER BY output").df()
barplot(d, "output", "n", "MCQ answer-letter balance", "Answer letter", "Count", "10_mcq_balance.png", rotate=0)

# 11. annotation density per image
d = con.execute("SELECT patch_id, COUNT(*) n FROM ben GROUP BY patch_id").df()
plt.figure(figsize=(8,5)); sns.histplot(d["n"], bins=40)
plt.title("Annotations per unique image pair"); plt.xlabel("Annotations per image"); plt.ylabel("Number of images")
plt.tight_layout(); plt.savefig("11_annotation_density.png"); plt.close()

# 12. geographic scatter (Section 8 code, repeated here for completeness)
# 13. type x category heatmap (Section 5 code)
# 14. bbox area histogram (needs parsed bbox_df from Section 7)
# 15. country x task, 16. season x climate — same pivot+heatmap pattern as Section 5
```

Every plot above already has a title/axis-label call built in per your requirement — run and interpret against your actual numbers, since I won't write interpretation text for numbers that don't exist yet.

---

## 14. Image-Level EDA (Sentinel-1 / Sentinel-2 pixel data)

**This cannot be determined from BigEarthNet.txt alone.** The Parquet file contains only `s1_name`/`patch_id` *identifiers*, not pixel arrays. Confirmed directly from the dataset card's own instructions: you must separately download the Sentinel-1/Sentinel-2 imagery from the BigEarthNet v2.0 site, then preprocess with `rico-hdl` into an LMDB of safetensors before any pixel-level statistic exists.

**Exact plan + code once you have the imagery locally:**
```python
# After: rico-hdl bigearthnet --bigearthnet-s1-dir <S1> --bigearthnet-s2-dir <S2> --target-dir Encoded-BigEarthNet
from ben_txt_datamodule import BENTxTDataset
import numpy as np

ds = BENTxTDataset(
    lmdb_file="Encoded-BigEarthNet/",
    metadata_file="BigEarthNet.txt.parquet",
    bands=("B04","B03","B02"),  # RGB for S2; adjust for VV/VH S1 bands
    img_size=120,
)
sample = ds[0]
img = sample["image_input"].numpy()

print("shape:", img.shape, "dtype:", img.dtype)
print("min/max:", img.min(), img.max())
print("mean/std:", img.mean(), img.std())
print("NaN count:", np.isnan(img).sum(), "Inf count:", np.isinf(img).sum())

import matplotlib.pyplot as plt
plt.hist(img.ravel(), bins=100); plt.title("Band value histogram — sample 0")
plt.savefig("band_hist_sample0.png")
```
Repeat across a sample of N images for per-band min/max/mean/std/percentiles; for Sentinel-1 do the same for VV/VH bands; for Sentinel-2 also compute band-to-band correlation matrices and build RGB (B04/B03/B02) and false-color NIR (B08/B04/B03) composites for visual sanity-checking.

---

## 15. ML/DL Interpretation — EDA Finding → Decision Chains

| EDA Finding | Why it matters | ML/DL Risk | Preprocessing Decision | Modeling Decision |
|---|---|---|---|---|
| ~20.6 annotations/image [COMPUTED] | 9.55M rows ≠ 9.55M independent visual samples | Inflated apparent dataset size; risk of the model learning text patterns instead of vision | Split by unique `patch_id`, not by row | Report both "annotation accuracy" and "per-image accuracy" separately |
| Multiple splits column present, leakage unverified | If a pair appears in train+test, evaluation is invalid | Falsely high test metrics | Run the Section 3 leakage query before any training | If leakage found, re-split by group before touching `test`/`bench` |
| 4 task types (binary/mcq/caption/bbox) | Each needs a structurally different output space | A single classification head cannot serve all 4 | Route inputs to task-specific preprocessing (tokenize question; for bbox, treat coordinates as structured targets) | Task-aware architecture: shared vision+text encoder, task-conditioned output heads (or a generative VLM that emits task-appropriate text) |
| Heavy question templating (Section 6, sample-confirmed) | High surface n-gram redundancy | Model may pattern-match phrasing rather than ground in pixels | Track template family per question (regex/cluster) so you can stratify eval by template, not just by category | Evaluate held-out on `bench` (manually verified, less templated) as the trustworthy signal, not raw `test` accuracy |
| `<point>`/`<ref>` literal tokens in text (sample-confirmed) | These are structural, not natural-language content | Standard tokenizers may fragment them inconsistently | Register `<point>`, `</point>`, `<ref>`, `</ref>` as special tokens | Model must learn to map these tokens to specific spatial locations — needs explicit spatial-grounding training signal, not just next-token prediction |
| S1 (SAR) + S2 (optical) pairing | Different physics, different noise characteristics, different value ranges | Naive channel-concatenation can let one modality dominate due to differing scale | Per-modality normalization (separate stats for SAR vs optical bands) | Dual-encoder fusion architecture (separate S1 and S2 encoders, late/cross-attention fusion), not a single shared conv stack |
| MCQ options embedded in `input`, not fixed vocabulary | `output`="b" means different things per question | A model could learn positional bias (e.g., "c" is right more often) | Check letter-position balance (Section 7) before trusting MCQ accuracy | Condition the answer head on the specific options text, don't use a fixed 4-way softmax over letters alone |
| Bounding-box outputs in free-text coordinate form | Needs parsing + validity checks (x1<x2, y1<y2) | Malformed generations are unusable without a parser/validator | Build a robust coordinate parser + reject/repair malformed outputs | Evaluate with IoU/mIoU, not exact string match |
| Data geography = 10 European countries only | ISRO's deployment context is India | Domain shift: crop types, terrain, season timing differ substantially from Indian conditions | None purely from this dataset — flag as a known gap | Present this explicitly as a limitation to judges; qualitatively test on any available Indian Sentinel imagery even without ground-truth labels |
| `bench` split is manually verified, smaller (1,082 pairs / 15,029 rows) | Higher-trust evaluation signal than auto-generated `test` | Over-trusting `test` numbers could mask real weaknesses | Keep `bench` untouched until final evaluation | Use `bench` as your primary "did this actually work" signal for the demo/judging narrative |

---

## 16. Preprocessing Pipeline (grounded in Section 15, not generic)

1. **Deduplication** — remove exact duplicate rows (Section 2); do NOT remove templated near-duplicates across different images (that's the dataset's legitimate structure, not a defect).
2. **Group-aware re-split verification** — confirm no `patch_id` crosses splits (Section 3); if provided splits are safe, keep them; if not, re-split by patch group.
3. **Text normalization** — lowercase-insensitive matching only for dedup/analysis, NOT for model input (case can carry meaning in captions); register `<point>`/`<ref>` as special tokens (Section 6).
4. **Tokenization** — subword tokenizer compatible with your chosen VLM backbone; verify `<point>`/`<ref>` survive tokenization as atomic units.
5. **Image normalization** — per-band, per-modality mean/std normalization (S1 SAR ≠ S2 optical statistics) — computed only once you have pixel access (Section 14).
6. **Bbox normalization** — outputs already appear to be in normalized [0,1] coordinates (from sample); verify this holds dataset-wide (Section 7) before assuming it, and reject/flag malformed strings rather than silently coercing them.
7. **Class balancing** — only after Section 7's real binary/MCQ balance numbers exist; if skewed, use class-weighted loss for binary/mcq, not naive oversampling (oversampling images multiplies visual redundancy on top of the existing ~20.6x annotation redundancy).
8. **Sequence-length handling** — set a max token length using the real P95 from Section 6, not a guessed constant; truncate/pad captions accordingly.
9. **Train/val/test prep** — build strictly at the `patch_id`-group level, carry all of a group's annotations together into one split.

---

## 17. Baseline Models

| Model | Input | Output | Loss | Metrics | Pros | Cons | Compute |
|---|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | question text only (no image) | binary/mcq answer | cross-entropy | accuracy, F1 | Fast, exposes text-only bias/shortcut baseline | Cannot do captioning/bbox; deliberately vision-blind — use ONLY to measure how much of the task is solvable from text alone (a critical diagnostic, not a real model) | CPU, minutes |
| Transformer text encoder (e.g. small BERT-family) fine-tuned on `input`→`output` | question text | binary/mcq answer | cross-entropy | accuracy, F1 | Stronger text-only ceiling check | Same vision-blindness | Single GPU, hours |
| CNN (ResNet-style) on S2 optical only | image only | fixed LULC-style label (not full QA) | cross-entropy | accuracy | Simple vision baseline | Ignores SAR entirely, ignores free-form questions | Single GPU |
| Pretrained remote-sensing encoder (e.g. a Sentinel-pretrained ViT/ResNet) | S1 or S2 patch | embedding for downstream head | task-dependent | task-dependent | Domain-appropriate pretraining beats generic ImageNet weights for satellite imagery | Needs sourcing/licensing the right checkpoint | Single GPU, fine-tuning |
| Multimodal fusion (S1 encoder + S2 encoder + text encoder + cross-attention) | image pair + question | task-appropriate output | task-specific (CE for binary/mcq, seq2seq CE for captioning, IoU-aware regression/CE for bbox) | accuracy/F1, BLEU/ROUGE, IoU per task | Matches the actual problem structure | Most complex, most compute | Multi-GPU or long single-GPU runs; LoRA fine-tuning of an existing VLM is the realistic hackathon path |

**Practical hackathon-scale recommendation:** LoRA/PEFT fine-tune an existing open VLM (already vision+language pretrained) on BigEarthNet.txt, rather than training a fusion architecture from scratch — matches your time budget and matches what's already been attempted publicly on this exact dataset (noted in the previous turn).

---

## 18. Final SatQuery AI Architecture (adjusted for EDA-grounded risks)

```
User Query (text)
        │
        ▼
Text Encoder ──────────────┐
                            │
Satellite Image(s) ─────┐  │
   (S1 SAR + S2 optical, │  │
    possibly T1/T2 pair) │  │
        │                │  │
        ▼                │  │
 S1 Encoder   S2 Encoder │  │
        │         │      │  │
        └────┬────┘      │  │
             ▼            │  │
     Cross-Modal Fusion ◄─┘  │
        (separate norm       │
        per modality —        │
        Section 15)           │
             │                │
             └───────┬────────┘
                      ▼
             Vision-Language Fusion
                      │
                      ▼
             Task Router (from `type`
             detected in the query, or
             a generative VLM emitting
             task-appropriate text)
                      │
        ┌─────┬───────┼────────┐
        ▼     ▼       ▼        ▼
     Binary  MCQ  Captioning  Bbox/Grounding
        │     │       │        │
        └─────┴───────┴────────┘
                      ▼
              Post-processing
        (bbox parsing+validation,
         MCQ letter grounding to
         option text, answer
         formatting)
                      │
                      ▼
                   Answer
```

Two deliberate additions versus your original sketch, both driven directly by Sections 5/7/15: (1) an explicit **task router** rather than one generic output head, because the 4 task types genuinely need different output formats; (2) an explicit **post-processing/validation** stage for bbox and MCQ, because free-text generation for structured tasks (coordinates, letter choices tied to specific options) needs parsing and sanity-checking, not blind trust in raw model output.

---

## 19. Evaluation

- **Binary:** Accuracy, Precision, Recall, F1 — plus a text-only baseline comparison (Section 17) to detect shortcut learning.
- **MCQ:** Accuracy — plus letter-position balance check (Section 7) to rule out positional bias inflating the score.
- **Captioning:** BLEU, ROUGE, and a semantic metric (BERTScore or similar) — n-gram metrics alone are known to reward template-matching, which this dataset structurally encourages (Section 6), so semantic scoring is not optional here.
- **Bounding box:** IoU, mean IoU, Accuracy@IoU-threshold (e.g., @0.5) — plus a "% outputs parseable at all" metric, since malformed coordinate strings are a real generative-model failure mode.
- **Stratified breakdowns:** country-wise, season-wise, climate-wise, category-wise — run all of these; given the geographic concentration (Section 8/9) and the India-deployment context, these breakdowns are where you'll find your most important limitations to disclose, not just nice-to-have plots.

---

## 20. Final SIH Report

### 1. Dataset Summary
464,044 co-registered Sentinel-1/Sentinel-2 image pairs over 10 European countries (2017–2018), with 9,553,962 text annotations across 4 task types (binary, MCQ, captioning, bounding box) and 11 finer categories, split into train/validation/test/bench (bench = 1,082 manually verified pairs / 15,029 annotations).

### 2. Top 10 EDA Findings (confirmed/computed only — not guessed)
1. ~20.6 annotations per image on average — dataset "size" in rows overstates true visual diversity by ~20x.
2. Bench split has fewer annotations per image (~13.9) than the dataset average, consistent with manual curation over template generation.
3. 4 structurally different task types exist in one file — no single output head can serve all of them.
4. `<point>`/`<ref>` are literal in-text tokens requiring special tokenizer handling.
5. MCQ answer letters do not map to a fixed meaning — must be conditioned on the specific question's options.
6. Bounding-box answers are free-text coordinate strings needing parsing and validity checks.
7. Underlying imagery is exclusively European (10 countries) — a real domain-shift risk against India-based deployment.
8. Heavy question templating is visible even in a 100-row sample — full-dataset duplication rate is unknown and must be measured (Section 6/11).
9. Whether the provided `split` column prevents image-level leakage is unverified — must be checked before trusting any reported accuracy (Section 3).
10. All pixel-level facts (band stats, quality, correlations) are entirely unknown until imagery is separately downloaded and preprocessed (Section 14).

### 3–16. Data Quality / Leakage / Imbalance / Text / Image / Multimodal / Preprocessing / Baselines / VLM / Evaluation / Risks / Limitations
Covered in full in Sections 2–19 above — not repeating here since they're not summarizable without fabricating the pending numbers.

### 16. Final Recommendation
Run the code in this document first (Sections 1–14) to replace every "unknown" with a real number, verify the split-leakage question before anything else, then proceed to a LoRA-fine-tuned VLM baseline (Section 17) evaluated primarily on `bench` (Section 19), with explicit disclosure of the European-imagery domain gap to your judges.

---

## WHAT I SHOULD TELL MY SIH TEAM

1. We have 464,044 real satellite image pairs, not 9.55 million — the 9.55M is annotations, and one image is asked about ~20 different ways on average. Don't oversell the dataset size.
2. Before we train anything, we must confirm no image pair appears in more than one split — if it does, our accuracy numbers will be fake-high and we need to re-split ourselves.
3. This dataset has 4 genuinely different task types (yes/no, multiple choice, captioning, bounding boxes) — we cannot build one generic output layer; we need a task-aware or fully generative architecture.
4. `<point>` and `<ref>` are literal tokens in the questions, not metadata — our tokenizer setup needs to handle them as special tokens or the model won't learn spatial grounding properly.
5. MCQ answers are letters, but the letter meaning changes per question — we can't treat this as a fixed 4-class problem; the model has to read the options.
6. Bounding-box answers are text strings we have to parse into coordinates ourselves, and we need to validate them (x1<x2, y1<y2) before computing IoU.
7. The dataset's underlying imagery is 100% European (10 countries, 2017-2018) — there's zero Indian imagery in the training data. We should say this openly to judges and try to qualitatively test on any Indian Sentinel scene we can get our hands on, even without ground truth.
8. Questions are heavily templated (we already saw this in a small sample) — the model might learn to pattern-match phrasing instead of truly understanding the image. We should trust the `bench` split (manually verified, 1,082 pairs) more than raw `test` accuracy.
9. The actual satellite pixel data is NOT in the file we've been analyzing — we still need to separately download BigEarthNet v2.0 imagery and preprocess it with `rico-hdl` before we can do any real vision-side EDA or training.
10. S1 (SAR) and S2 (optical) have completely different physics and value ranges — we need separate normalization per modality, not one shared preprocessing step.
11. We don't yet know the real class balance (binary yes/no split, MCQ letter distribution, category imbalance) — that's the first script to run, not something to assume going in.
12. Given hackathon time constraints, LoRA/PEFT fine-tuning an existing open vision-language model is more realistic than training a fusion architecture from scratch.
13. Our evaluation plan should break results down by country/season/climate/category, not just report one overall accuracy — that's where real weaknesses (and honest talking points for judges) will show up.
14. Every number in this document that isn't explicitly labeled "confirmed" or "computed" is unknown until someone on the team actually runs the DuckDB/Polars code — nobody should quote an imbalance ratio or duplication rate in the pitch deck without having run it first.
15. The single biggest risk to catch early is data leakage across splits — that one check (Section 3) determines whether every other metric we report is trustworthy.
