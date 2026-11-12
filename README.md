# 🌊 OASIS — Oil-spill Attribution & Satellite Intelligence System
### Marine Forensics Platform for the Indian Coast Guard

> Semester 5 Mini Project · Image and Video Processing Lab (CSET344)  
> Department of Computer Science & Engineering · **Team Alpha**

---

## What is OASIS?

OASIS is a full-stack geospatial intelligence platform that:

1. **Detects** marine hydrocarbon slicks from SAR (Synthetic Aperture Radar) satellite imagery using a trained U-Net deep learning model
2. **Verifies** the detection radiometrically using an EfficientNet-B0 forensic verifier (distinguishing real oil from sea clutter look-alikes)
3. **Traces** the spill's probable origin backwards in time using Lagrangian leeway physics (ocean current + wind drift + Coriolis deflection)
4. **Forecasts** the slick's future dispersion cone (12–24 hours ahead)
5. **Identifies** responsible vessels by correlating real AIS (Automatic Identification System) vessel tracks against the computed origin spatio-temporal envelope
6. **Generates** legally defensible forensic PDF dossiers with SHA-256 chain-of-custody hashes

Every conclusion is deterministically traceable — no black-box attribution.

---

## Screenshots

> Upload a SAR image → get spill boundary, drift hindcast, and suspect vessels plotted on an interactive map in seconds.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **AI Model 1** | U-Net (PyTorch) — SAR oil spill segmentation |
| **AI Model 2** | EfficientNet-B0 (PyTorch) — Radiometric forensic verifier |
| **Physics Engine** | Lagrangian leeway advection (custom Python) |
| **AIS Query Engine** | DuckDB — spatiotemporal CSV query over 58M+ rows |
| **Backend API** | FastAPI + Uvicorn (Python 3.12+) |
| **Frontend** | React 19, TypeScript, Vite |
| **Map** | MapLibre GL JS |
| **3D Globe** | Three.js |
| **Charts** | Recharts |
| **State** | Zustand + TanStack Query |
| **PDF Reports** | ReportLab |

---

## Project Structure

```
Oasis-main/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app entrypoint
│   │   ├── schemas.py               # Pydantic data contracts
│   │   ├── api/v1/endpoints.py      # REST API route handlers
│   │   └── core/
│   │       ├── segmentation.py      # Layer 3: U-Net SAR segmentation
│   │       ├── lookalike_verifier.py# Layer 3: EfficientNet forensic verifier
│   │       ├── geometry.py          # Layer 3: OpenCV metric geometry extraction
│   │       ├── drift_engine.py      # Layer 4: Lagrangian leeway physics engine
│   │       ├── ais_service.py       # Layer 5: DuckDB AIS spatiotemporal query
│   │       ├── attribution.py       # Layer 5: Multi-factor vessel scoring
│   │       ├── metocean_service.py  # Live metocean data fetcher
│   │       ├── model_downloader.py  # HuggingFace model auto-downloader
│   │       └── scenarios.py        # Pre-built demo scenarios
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/                   # SpillLab, Dashboard pages
│   │   ├── components/              # UI components
│   │   ├── map/                     # MapLibre GL map layer
│   │   ├── globe/                   # Three.js 3D globe
│   │   ├── store/                   # Zustand state management
│   │   ├── api/                     # API client (TanStack Query)
│   │   └── types/                   # TypeScript type definitions
│   └── package.json
├── ais dataset/                     # AIS CSV files (place here)
│   └── ais-YYYY-MM-DD.csv
└── Test Images/                     # Sample SAR test images
    └── IMG_01.jpeg … IMG_12.jpeg
```

---

## 7-Layer Pipeline Architecture

```
┌─────────────────────────────────────────────────────┐
│  Layer 1 · Data Ingestion                           │
│  Sentinel-1 SAR · ERA5 Wind · CMEMS Currents · AIS  │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  Layer 2 · Preprocessing                            │
│  512×512 Tiling · dB Normalization · AIS Indexing   │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  Layer 3 · Dual-AI Perception Engine                │
│  Model 1: U-Net SAR Segmentation (mask + heatmap)   │
│  Task 2:  OpenCV Geometry (area km², centroid, θ)   │
│  Model 2: EfficientNet Forensic Verifier (oil/noise)│
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  Layer 4 · SlickTrace Physics Drift Engine          │
│  V_slick = V_current + 0.03·V_wind + Coriolis       │
│  ← Hindcast 6 h to probable origin P₀              │
│  → Forecast 12 h dispersion cone                   │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  Layer 5 · AIS Correlation & Vessel Attribution     │
│  DuckDB spatiotemporal query around P₀             │
│  Multi-factor scoring: CPA · heading · speed anomaly│
│  Dark vessel protocol if AIS = 0                   │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  Layer 6 · Forensic Evidence & Integrity            │
│  SHA-256 chain-of-custody · confidence matrix       │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  Layer 7 · Command Delivery                         │
│  React 19 Tactical UI · FastAPI REST · PDF Dossier  │
└─────────────────────────────────────────────────────┘
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+
- ~5 GB disk space (for PyTorch + AI model weights)

### 1 — Clone & enter the project

```bash
git clone https://github.com/mittalyash616/oasis-platform.git
cd oasis-platform
```

### 2 — Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> AI model weights are auto-downloaded from HuggingFace on first run (`KrrishDangi/OASIS-models`).

### 3 — AIS Dataset

Place AIS CSV files inside the `ais dataset/` folder at the project root. Files must follow this column schema:

```
mmsi, base_date_time, longitude, latitude, sog, cog, heading,
vessel_name, imo, call_sign, vessel_type, status, length, width,
draft, cargo, transceiver
```

> Compatible with MarineCadastre / NOAA AIS export format. The system uses DuckDB to query across all CSVs simultaneously — no import step required.

### 4 — Frontend setup

```bash
cd frontend
npm install --cache /tmp/npm-cache
```

### 5 — Run

Open two terminals:

**Terminal 1 — Backend:**
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev -- --host
```

| Service | URL |
|---------|-----|
| Tactical UI | http://localhost:5173 |
| REST API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |

---

## API Reference

### `POST /api/v1/analyze`

Runs the full 7-layer pipeline on a SAR image.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | `UploadFile` | SAR image (JPEG / PNG / GeoTIFF) |
| `approx_lat` | `float` | Observation latitude |
| `approx_lon` | `float` | Observation longitude |
| `observation_time` | `datetime` | UTC timestamp of satellite pass |
| `wind_speed_ms` | `float` | Wind speed in m/s (optional — fetched live if omitted) |
| `wind_dir_from` | `float` | Wind direction from (°) |
| `current_speed_ms` | `float` | Surface current speed in m/s |
| `current_dir_to` | `float` | Surface current direction to (°) |

**Response:** `SpillAnalysisResponse` — includes geometry, drift, ranked vessels, confidence scores.

### `GET /api/v1/metocean?lat=&lon=`
Fetches live wind and ocean current data for a coordinate.

### `GET /api/v1/scenarios/{id}`
Returns pre-built demo analysis. IDs: `mumbai_high`, `kutch_dark_vessel`, `bengal_lookalike`.

### `POST /api/v1/report/generate`
Generates a court-admissible forensic PDF from an analysis result.

---

## Vessel Attribution Scoring

Each candidate vessel in the AIS search radius is scored by:

| Factor | Weight | Description |
|--------|--------|-------------|
| **CPA Distance** | 40% | Closest Point of Approach to computed origin |
| **Heading Alignment** | 25% | Vessel COG vs. slick elongation axis |
| **Speed Anomaly** | 25% | Sudden drop below 6 kt during transit (tank-washing indicator) |
| **Area Presence** | 10% | Baseline score for being in the search envelope |

Confidence levels: `HIGH` (score ≥ 0.75, CPA ≤ 3 km) · `MEDIUM` (≥ 0.50) · `LOW` (< 0.50)

---

## Demo Scenarios

Three pre-loaded scenarios are available without uploading any image:

| Scenario | Location | AIS Status | Description |
|----------|----------|-----------|-------------|
| `mumbai_high` | Mumbai High, Arabian Sea | ✅ Available | Crude tanker discharge, high confidence attribution |
| `kutch_dark_vessel` | Gulf of Kutch | ⚠️ Unavailable | AIS-off dark vessel, intercept coordinates issued |
| `bengal_lookalike` | Bay of Bengal | ✅ Available | Sea clutter look-alike, correctly rejected by Model 2 |

---

## Confidence Breakdown

The UI displays a **5-factor confidence score**:

| Factor | Source |
|--------|--------|
| Satellite Quality | SAR image resolution and dual-pol availability |
| Segmentation Quality | U-Net model output confidence |
| Environmental Data | Metocean data freshness and coverage |
| AIS Coverage | AIS data density in the search area |
| Evidence Consistency | Multi-factor agreement across all pipeline layers |

---

## Forensic PDF Report

Generated via `POST /api/v1/report/generate`, the PDF includes:

- Spill geometry map with centroid and bounding box
- Drift hindcast and forecast trajectory
- Ranked vessel table with attribution scores
- SHA-256 chain-of-custody hash
- Confidence breakdown matrix
- NOS-DCP compliance statement

---

## Design Principles

- **No black-box attribution** — every conclusion traces to physics equations and auditable data
- **Dark vessel honesty** — if AIS is absent, the system flags "Attribution Unavailable" and outputs aerial intercept coordinates rather than inventing a suspect  
- **Modular pipeline** — each layer has a strictly typed Pydantic data contract; components can be independently tested or swapped
- **Legal defensibility** — output format is aligned with ICG / NOS-DCP forensic standards

---

## License

This project was developed as a Semester 5 Mini Project for the Image and Video Processing Lab (CSET344).  
© 2026 Team Alpha. All rights reserved.
