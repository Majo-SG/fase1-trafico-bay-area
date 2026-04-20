# Phase 1 — Traffic Data Preprocessing
### Project: Unified Prediction Model (Congestion + Incidents)

---

## Project Description

This repository contains the work for **Phase 1** of a research project aimed at building a multimodal spatio-temporal architecture capable of simultaneously predicting vehicle congestion and traffic incidents using Large Language Models (LLMs).

The central hypothesis is that a fundamental gap exists in current systems: existing models either predict numerical variables (speed/flow) **or** analyze incident text, but none process both as a unified sequence. This project proposes to fill that gap.

---

## Datasets

### PEMS-BAY
- **Source:** Caltrans Performance Measurement System (PeMS)
- **Download:** [Zenodo — zenodo.org/records/5146275](https://zenodo.org/records/5146275)
- **Period:** January – June 2017
- **Sensors:** 325 sensors across the San Francisco Bay Area
- **Frequency:** Every 5 minutes
- **Variable:** Traffic speed in miles per hour (mph)
- **Reference:** Li et al., DCRNN, ICLR 2018

### US Accidents
- **Source:** Moosavi et al., Ohio State University
- **Download:** [Kaggle — kaggle.com/datasets/sobhanmoosavi/us-accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- **Total records USA:** 7,728,394
- **Filtered to Bay Area 2017:** 14,243
- **Key variables:** text description, severity (1–4), GPS coordinates, weather conditions

> **Note:** Raw data files are not included in this repository due to their size (PEMS-BAY: 82 MB, US Accidents: 2.9 GB). Download them from the links above and place them in a local `data/` folder.

---

## Preprocessing Steps

### 1. Load and clean
Loaded `PEMS-BAY.csv` with 52,116 timestamps and 325 sensors. Verified absence of NaN values (0 found). Applied forward fill and backward fill as standard cleaning protocol.

### 2. Time Series Resampling
Converted the time series from 5-minute resolution to hourly resolution by averaging, reducing from 52,116 to 4,344 rows.

### 3. Heuristic function f(v) → T
Implemented the following velocity → text description mapping function:

```
v < 20 km/h           → "Severe congestion reported at node i."
20 ≤ v ≤ 60 km/h      → "Moderate traffic flow at node i."
v > 60 km/h           → "Free flow traffic at node i."
∆v > 40 km/h in 5 min → "Sudden slowdown, possible accident."
```

### 4. Temporal gradient detector
Implemented a detector for sudden speed drops: if the difference between two consecutive timestamps exceeds 40 km/h, the possible accident label is assigned. **2,156 events** detected (4.96% of total).

### 5. Alignment with US Accidents
The 14,243 real Bay Area accidents were aligned with the time series by date and hour, integrating the real text description when available, and "Normal traffic flow" when not.

### 6. Strict chronological split
To avoid **temporal data leakage**, the split was done sequentially (never randomly):

| Set | Proportion | Rows | Period |
|---|---|---|---|
| `train.csv` | 70% | 30,441 | Jan 2017 → Apr 2017 |
| `val.csv` | 10% | 4,348 | Apr 2017 → May 2017 |
| `test.csv` | 20% | 8,699 | May 2017 → Jun 2017 |

Zero overlap between splits and 0 NaN / 0 NaT verified in each partition.

### 7. Adjacency matrix
The `adj_mx.pkl` matrix of 325×325 sensors represents the road network with weights calculated using **Gaussian Kernel Weighting**:

```
w_ij = exp(−d²/σ²)   if d < 30 km
w_ij = 0              if d ≥ 30 km
```

Result: 2,694 active connections with 2.55% density.

---

## Master Table Structure

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Date and time (index) |
| `speed_mph` | float | Speed in miles per hour |
| `speed_kmh` | float | Speed in kilometers per hour |
| `delta_v_kmh` | float | Speed variation from previous timestamp |
| `incident_description` | string | Text description of traffic state |
| `sensor_id` | string | Reference sensor identifier |

---

## How to Reproduce

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scipy h5py tables
```

### Execution
```bash
mkdir data
# Copy PEMS-BAY.csv, adj_mx_bay.pkl, PEMS-BAY-META.csv, US_Accidents_March23.csv into data/
python fase1_pipeline_REAL.py
```

---

## Project Phases

### ✅ Phase 1 — Data Engineering and Preprocessing
Construction of the unified database that will feed the model.

### 🔲 Phase 2 — Numerical Tokenization and Semantic Embeddings
- Uniform Quantization or VQ-VAE to convert speed into discrete tokens
- Semantic embeddings of incident descriptions using a pretrained language model
- Multimodal token-package combining both modalities per timestamp

### 🔲 Phase 3 — LLM Architecture and Fine-tuning
- Backbone: Llama 3 (8B) or Mistral 7B
- Low-Rank Adaptation (LoRA) for efficient fine-tuning
- Partially Frozen Attention for spatio-temporal dependency capture

### 🔲 Phase 4 — Evaluation and Results
Evaluation against state-of-the-art baselines.
Metrics: MAE, RMSE, MAPE at 15, 30 and 60-minute horizons.

---

## References

- Li, Y. et al. (2018). *Diffusion Convolutional Recurrent Neural Network.* ICLR 2018.
- Moosavi, S. et al. (2019). *A Countrywide Traffic Accident Dataset.* arXiv:1906.05409.
- Liu, C. et al. (2024). *Spatial-Temporal Large Language Model for Traffic Prediction.* arXiv:2401.10134.
- Liu, L. et al. (2024). *How Can Large Language Models Understand Spatial-Temporal Data?* arXiv:2401.14192.
- Yang, X. et al. (2025). *LLeCaT: LLM Enhanced Causality-aware Traffic Accidents.* IEEE TITS.
- Li, Z. et al. (2025). *Open Spatio-Temporal Foundation Models for Traffic Prediction.* ACM TIST.

---

## Team
Research project — Systems Engineering / Data Science
San Francisco Bay Area, California · PEMS-BAY Dataset 2017
