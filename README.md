# Eaton Fire - Building Damage Assessment

Building damage assessment for the 2025 Eaton Fire in Altadena, California using Sentinel-2 satellite imagery and machine learning.

## Overview

This pipeline processes Sentinel-2 satellite imagery to detect building damage from wildfires using a hybrid approach combining clustering and XGBoost.

**Key features:**
- Sentinel-2 data download and processing
- Building footprint extraction
- Hybrid ML: Clustering + XGBoost
- Interactive web map
- GeoJSON export

## Results

| Metric | Value |
|--------|-------|
| Buildings Analyzed | 37,291 |
| Damaged Buildings | 10,607 (28.4%) |
| High Severity | 10,578 |
| Found by Clustering | 6,034 |
| Found by ML Refinement | 4,573 |

## Pipeline Steps

1. **Download Sentinel-2 data** - Before and after fire imagery
2. **Compute spectral indices** - dNBR, NDVI, SWIR, Brightness
3. **Extract building features** - From building footprints
4. **Create seed labels** - Low threshold classification
5. **Apply clustering** - K-Means and K-NN
6. **Refine with XGBoost** - Train on uncertain cases
7. **Generate outputs** - CSV, GeoJSON, HTML map

## Installation

```bash
# Clone repository
git clone https://github.com/joshscrabeck/Eaton-Fire-Damage-Assessment.git
cd Eaton-Fire-Damage-Assessment

# Create environment
conda create -n fire-damage python=3.9
conda activate fire-damage

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Download data
python src/data_downloader_10bands.py

# Run damage assessment
python src/hybrid_indices_ML_KMeans.py

# Generate web map
python src/eaton_web_map.py
```

## Project Structure

```
Eaton-Fire-Damage-Assessment/
├── src/
│   ├── data_downloader_10bands.py
│   ├── hybrid_indices_ML_KMeans.py
│   └── eaton_web_map.py
├── data/
│   └── sentinel2/
│       ├── raw/
│       └── processed/
├── output/
├── requirements.txt
├── README.md
└── LICENSE
```

## Output Files

| File | Description |
|------|-------------|
| `building_damage_*.csv` | Full damage data |
| `damaged_buildings_*.geojson` | Damaged buildings for GIS |
| `*_analysis.png` | Results visualization |
| `eaton_fire_damage_map.html` | Interactive map |

## Methodology

**Spectral Indices:**
- dNBR: Measures fire severity
- NDVI Change: Vegetation loss
- SWIR Ratio Change: Heat signatures
- Brightness Change: General damage

**Classification:**
- High (3): Complete destruction
- Moderate (2): Significant damage
- Low (1): Minor damage
- Undamaged (0): No damage

**Feature Importance (XGBoost):**
- dNBR 75th Percentile: 95.4%
- dNBR Mean: 1.8%
- dNBR Max: 1.2%
- NDVI Min: 0.6%
- SWIR Mean: 0.2%

## Limitations

- 10m resolution limits building-level accuracy
- Mixed pixels affect small buildings
- Shadows and vegetation can cause false positives
- Cloud cover may obscure damage

## License

MIT License

## Acknowledgments

- Copernicus Sentinel-2
- OpenStreetMap
- Overture Maps

---

**Note**: Research-grade tool. Validate results with ground truth.
