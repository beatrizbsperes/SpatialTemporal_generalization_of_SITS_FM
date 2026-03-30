# Spatial Temporal Generalization of Time Series Foundation Model Embeddings for Crop Classification

This project investigates how well embeddings from a **time series remote sensing foundation model ([Tessera](https://github.com/tessera-rs/tessera))** generalize across **space** (unseen geographic regions) and **time** (unseen years) for crop type classification in Europe.

The labeled dataset is sourced from **[EuroCrops](https://github.com/maja601/EuroCrops) (v02)**, focusing on **3 crop types** across multiple European countries.

---

## Research Objectives

1. **Spatial Generalization** — Train a classifier on embeddings from points in 4 countries, evaluate on 2 held-out countries. Does Tessera produce geographically transferable representations?
2. **Temporal Generalization** — Train on embeddings from a subset of years (2018–2023), evaluate on held-out years. Does the model capture stable crop phenology across growing seasons?

---

## Data

### Ground Truth Labels
- **Source**: EuroCrops v02
- **Coverage**: ~500 labeled field points per crop per country (latitude/longitude coordinates)
- **Classes**: 3 crop types

### Remote Sensing Inputs

| Sensor | Bands / Indices |
|--------|----------------|
| **Sentinel-1 (S1)** | `VV`, `VH`, `VV_VH_ratio`, `RVI` |
| **Sentinel-2 (S2)** | `B2`, `B3`, `B4`, `B5`, `B6`, `B7`, `B8`, `B8A`, `B11`, `B12`, `NDVI`, `NDWI`, `NDRE`, `EVI` |

Each labeled point has **all available S1 and S2 observations across a full year** — forming a **multi-sensor time series** per point.

### Tessera Embeddings
- **Temporal coverage**: 2018–2023
- **Granularity**: Point-pixel embeddings extracted per location per year
- Stored under `_data/embeddings_dir/`

---

## Project Structure

```
SPATIALTEMPORAL_GENERALI.../
│
├── _data/
│   ├── embeddings_dir/          # Tessera embeddings per point × year
│   ├── eurocrops2/              # Raw EuroCrops v02 labels and geometries
│   └── exports/                 # Processed outputs and intermediate datasets
│
├── spatial_analysis/
│   ├── 2_extract_embeddings_spatial.ipynb   # Extract Tessera embeddings for spatial split
│   └── 3_get_images_spatial.ipynb           # Retrieve S1/S2 imagery for spatial points
│
├── temporal_analysis/
│   ├── 2_extract_embeddings_temporal.ipynb  # Extract Tessera embeddings for temporal split
│   └── 3_get_images_temporal.ipynb          # Retrieve S1/S2 imagery for temporal points
│
├── 1_dataset_explorer.ipynb         # EDA: label distribution, time series plots, coverage maps
├── 4_embeddings_analysis.ipynb      # Embedding space analysis (PCA, UMAP, cluster inspection)
├── 5_spatial_generalization.ipynb   # Train/evaluate spatial generalization experiments
├── 6_temporal_generalization.ipynb  # Train/evaluate temporal generalization experiments
├── 7_generate_embedding.ipynb       # Utility: generate embeddings for new points/years
│
├── pyproject.toml
├── uv.lock
├── .env
└── .gitignore
```

---

## Methodology

### Feature Engineering from Raw Time Series

For each labeled point, the raw S1/S2 time series is processed to extract features that capture **crop seasonality**:

- **Temporal statistics** per band/index (mean, std, min, max, percentiles)
- **Phenological aggregates** — growing season statistics, peak greenness timing
- **Time-series-aware features** — derived from the irregular temporal sampling of Sentinel observations across the year

### Tessera Embeddings Pipeline

1. For each labeled point and year, a **patch of S1+S2 imagery** is retrieved (notebooks `3_get_images_*.ipynb`)
2. The Tessera foundation model produces a **fixed-length embedding** per point-year (notebooks `2_extract_embeddings_*.ipynb`)
3. Embeddings are analyzed for **separability by crop class** across different spatial and temporal domains (`4_embeddings_analysis.ipynb`)

### Generalization Experiments

**Spatial** (`5_spatial_generalization.ipynb`):
- Train countries: 4 European countries
- Test countries: 2 held-out European countries
- Classifier trained on Tessera embeddings; evaluated zero-shot on unseen geographies

**Temporal** (`6_temporal_generalization.ipynb`):
- Train years: subset of 2018–2023
- Test years: held-out year(s)
- Evaluates whether embeddings encode stable phenological patterns independent of acquisition year

---

## Getting Started

### Requirements

```bash
# Install dependencies using uv
uv sync
```

### Recommended Notebook Order

```
1_dataset_explorer.ipynb
    → spatial_analysis/2_extract_embeddings_spatial.ipynb
    → spatial_analysis/3_get_images_spatial.ipynb
    → temporal_analysis/2_extract_embeddings_temporal.ipynb
    → temporal_analysis/3_get_images_temporal.ipynb
4_embeddings_analysis.ipynb
5_spatial_generalization.ipynb
6_temporal_generalization.ipynb
```

---

## Key Questions Being Explored

- Do Tessera embeddings form **crop-discriminative clusters** without any fine-tuning?
- How much does classification accuracy **degrade across countries** never seen during training?
- How much does classification accuracy **degrade across years** never seen during training?
- Is there an advantage of using **foundation model embeddings** over hand-crafted time series features (S1/S2 statistics)?

---
