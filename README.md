AI-powered system for satellite image analysis that classifies land use and detects deforestation using CNN and change detection techniques. Includes Environmental Impact Scoring (EIS) and AI-generated insights for real-world environmental monitoring.
# 🌿 AI Land Use & Deforestation Analysis System
https://ai-land-use-and-deforestration-analysis.onrender.com

> An AI-powered satellite image analysis dashboard for detecting deforestation, classifying land use, and generating environmental impact reports — built with TensorFlow, OpenCV, and Streamlit.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training the Model](#training-the-model)
- [Running the Dashboard](#running-the-dashboard)
- [How It Works](#how-it-works)
- [Dashboard Sections](#dashboard-sections)
- [AI Extensions](#ai-extensions)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)
- [Troubleshooting](#troubleshooting)

---

## Overview

This system analyses pairs of satellite images taken at different points in time to:

- **Classify** land into five categories: Forest, Water, Urban, Agriculture, and Barren
- **Detect** deforestation, urban expansion, and water body changes
- **Quantify** environmental loss with area statistics and an Environmental Impact Score (EIS)
- **Report** findings in natural language with risk classification and recommended actions
- **Chat** with an AI assistant (Gemini) about the analysis results

Upload a before-and-after pair of satellite images and get a complete environmental analysis in seconds.

---

## Features

| Feature | Description |
|---|---|
| 🗺️ Land Use Classification | CNN-based pixel classification into 5 land cover types |
| 🔥 Change Detection | Pixel-level deforestation and urban expansion detection |
| 🌡️ Heatmap Visualisation | Change intensity heatmap with colour-coded severity |
| 📊 Plotly Analytics | Bar, gauge, pie, and trend charts for area statistics |
| 📋 Intelligent Report | Natural-language findings, risk badge, and EIS score |
| 💬 Gemini AI Chatbot | Conversational AI for deeper analysis (optional) |
| 🌱 Carbon Estimation | Forest loss translated into estimated carbon impact |
| 🦜 Species Risk Panel | Biodiversity risk from habitat loss |
| 📍 Location Mapping | Geographic context panel |
| ⚠️ Anomaly Alerts | Automated alerts for critical change events |
| ⬇️ HTML Report Export | Downloadable full analysis report |

---

## Project Structure

```
AI-LAND-USE-AND-DEFORESTATION-ANALYSIS/
│
├── app.py                  # Main Streamlit dashboard
├── train_model.py          # Model training script
├── download_dataset.py     # Dataset downloader & organiser
│
├── preprocess.py           # Image preprocessing utilities
├── predict.py              # Land use prediction pipeline
├── detect_change.py        # Deforestation change detection
├── report_generator.py     # Report generation (charts + text)
│
├── gemini_analyst.py       # Gemini AI chatbot integration
├── ai_extensions.py        # Carbon, species, alerts, trend panels
│
├── data/                   # Training dataset (auto-created)
│   ├── Forest/
│   ├── Water/
│   ├── Urban/
│   ├── Agriculture/
│   └── Barren/
│
├── models/                 # Saved model files (auto-created after training)
│
├── reports/                # Generated HTML/JSON reports (auto-created)
│
└── requirements.txt        # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/SravyaMummana2006/AI-LAND-USE-AND-DEFORESTRATION-ANALYSIS.git
cd AI-LAND-USE-AND-DEFORESTRATION-ANALYSIS
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include:

```
streamlit
tensorflow
opencv-python
numpy
pandas
pillow
plotly
python-dotenv
google-generativeai   # optional — for Gemini AI chatbot
```

### 4. (Optional) Configure Gemini API Key

To enable the AI chatbot and Gemini-powered insights, create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_free_api_key_here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey). The app runs fully without this — the chatbot section is simply hidden.

---

## Dataset Setup

Run the dataset downloader to prepare training data. Two options are available:

```bash
python download_dataset.py
```

You will be prompted to choose:

**Option 1 — Download EuroSAT RGB (Real Satellite Images)**
- Downloads ~90 MB from the public EuroSAT dataset (Zenodo, no login needed)
- Extracts and organises 100 images per class into `data/`
- Cleans up the large zip file automatically
- Falls back to synthetic generation if download fails

**Option 2 — Generate Synthetic Dataset (No Internet Required)**
- Generates 100 colour-pattern images per class instantly
- Realistic HSV colour profiles per land type
- Useful for testing the pipeline without an internet connection

After setup, the `data/` folder will look like:

```
data/
├── Forest/        100 images
├── Water/         100 images
├── Urban/         100 images
├── Agriculture/   100 images
└── Barren/        100 images
```

### EuroSAT Class Mapping

The downloader maps EuroSAT's 10 original classes to the 5 project classes:

| EuroSAT Class | Project Class |
|---|---|
| Forest | Forest |
| River, SeaLake | Water |
| Residential, Industrial, Highway | Urban |
| AnnualCrop, PermanentCrop | Agriculture |
| HerbaceousVegetation, Pasture | Barren |

---

## Training the Model

Once the dataset is ready, train the CNN classifier:

```bash
python train_model.py --data_dir ./data --epochs 30
```

**Common flags:**

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `./data` | Path to the training dataset |
| `--epochs` | `30` | Number of training epochs |
| `--batch_size` | `32` | Training batch size |
| `--model_type` | `custom_cnn` | `custom_cnn` or `mobilenet` |
| `--img_size` | `256` | Input image size in pixels |

The trained model is saved to the `models/` directory. **The model must be trained before running the dashboard.** If you run the app without a trained model you will see:

```
Model not found. Please train the model first.
```

---

## Running the Dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` in your browser.

**Quick Start:**
1. Upload an **old (baseline) satellite image** in the sidebar
2. Upload a **new (recent) satellite image** in the sidebar
3. Click **🔍 Run Analysis**
4. Explore results across all dashboard sections

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

---

## How It Works

### Analysis Pipeline

When you click **Run Analysis**, the following steps execute in sequence:

```
Upload Images
     │
     ▼
1. Preprocess         — Normalise, resize, and prepare images for the model
     │
     ▼
2. Predict (Old)      — Run CNN land use classification on baseline image
     │
     ▼
3. Predict (New)      — Run CNN land use classification on recent image
     │
     ▼
4. Compare            — Diff area statistics between old and new predictions
     │
     ▼
5. Change Detection   — OpenCV pixel-level deforestation & urban change detection
     │
     ▼
6. Report Generation  — Compute EIS, risk level, charts, and natural-language findings
     │
     ▼
7. AI Extensions      — Carbon estimates, species risk, alerts, trend panels
     │
     ▼
8. (Optional) Gemini  — Build context and initialise AI chatbot
```

### Land Use Classes

| Class | Description |
|---|---|
| 🌳 Forest | Dense tree cover, woodland |
| 💧 Water | Rivers, lakes, reservoirs |
| 🏙️ Urban | Built-up areas, roads, infrastructure |
| 🌾 Agriculture | Cropland, farmland |
| 🏜️ Barren | Bare soil, grassland, scrubland |

### Environmental Impact Score (EIS)

The EIS is a composite score from 0 to 100:

| Score | Meaning |
|---|---|
| 0–24 | Minimal impact |
| 25–49 | Moderate impact |
| 50–74 | High impact |
| 75–100 | Critical / severe impact |

It factors in forest loss percentage, urban expansion rate, water body change, and the number of detected change regions.

---

## Dashboard Sections

### 1. Sidebar
Upload controls, model settings, change sensitivity slider, and the Run Analysis button.

### 2. Image Viewer
Side-by-side display of the uploaded old and new satellite images.

### 3. Land Use Classification
Colour-coded overlay maps for both images showing predicted land use per region, with class probability tables and area coverage breakdowns.

### 4. Change Detection
- Annotated highlight image showing forest loss (red) and urban expansion (orange)
- Change intensity heatmap (blue → green → yellow → red)
- Detected change region table with area and centroid coordinates
- Six key metric cards: forest loss %, forest gain %, urban expansion %, water change %, total changed %, and number of changed regions

### 5. Analytics & Charts
Four Plotly tabs:
- **Coverage Comparison** — grouped bar chart of old vs new land coverage
- **Environmental Indicators** — forest loss gauge and EIS gauge
- **Change Breakdown** — pie chart of change composition
- **Net Class Change** — horizontal bar showing per-class gain/loss

### 6. Intelligent Report
- Risk badge (Low / Moderate / High / Critical) with colour coding
- EIS progress bar
- Executive summary paragraph
- Detailed forest, urban, and water findings
- Per-class change bullet points
- Recommended action block
- Full metrics summary table
- Download button for the HTML report

### 7. AI Chatbot (Gemini — Optional)
Ask natural-language questions about the analysis results. Requires a Gemini API key in `.env`.

---

## AI Extensions

These panels are automatically rendered after every analysis:

| Panel | Description |
|---|---|
| ⚠️ Alerts | Critical change event notifications |
| 🌱 Carbon Impact | Estimated CO₂ from detected forest loss |
| 📈 Trend Analysis | Projected land use trends based on current change rates |
| 👁️ Vision Insights | Computer vision feature summary of both images |
| 📍 Location Context | Geographic and environmental context panel |
| 🦜 Species Risk | Biodiversity risk assessment from habitat loss |

---

## Configuration

### Sidebar Settings

| Setting | Options | Description |
|---|---|---|
| Model Type | Custom CNN, MobileNetV2 Transfer | Classification architecture |
| Change Sensitivity | 10–60 (default 30) | Pixel diff threshold for change detection. Lower = more sensitive |
| Save HTML Report | On/Off | Export report after analysis |

### Environment Variables (`.env`)

```env
GEMINI_API_KEY=your_key_here   # Optional — enables AI chatbot
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Dashboard | Streamlit |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Data Processing | NumPy, Pandas |
| Visualisation | Plotly |
| Image Handling | Pillow (PIL) |
| AI Chatbot | Google Gemini API |
| Dataset | EuroSAT RGB (Zenodo) |
| Styling | Custom CSS (DM Sans, DM Serif Display) |

---

## Troubleshooting

**`Model not found` error when running the app**
The model has not been trained yet. Run:
```bash
python train_model.py --data_dir ./data --epochs 30
```

**`Analysis failed` error**
Check that both images are valid satellite images in a supported format (`.jpg`, `.png`, `.tif`). Very small images (under 64×64 px) may cause issues.

**Gemini chatbot not appearing**
Add your Gemini API key to a `.env` file and restart the app:
```env
GEMINI_API_KEY=your_key_here
```

**Download failed in `download_dataset.py`**
The EuroSAT download requires an internet connection to `madm.dfki.de`. If it times out, choose option 2 (synthetic dataset) instead.

**Streamlit port already in use**
```bash
streamlit run app.py --server.port 8502
```

---

## Author

AI Land Use Analysis System — built with TensorFlow · OpenCV · Streamlit

> For deforestation research, environmental monitoring, urban planning analysis, and satellite remote sensing education.
