# 🌫️ Air Quality Predictor — ML Web Application

> Predict the Air Quality Index (AQI) from pollutant sensor readings using a trained Machine Learning model, served through a Flask API with an interactive web UI.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [How It Works](#how-it-works)
- [ML Model Details](#ml-model-details)
- [API Reference](#api-reference)
- [AQI Categories](#aqi-categories)
- [Dataset](#dataset)

---

## Overview

This project is an end-to-end **Machine Learning web application** that predicts Air Quality Index (AQI) based on 8 environmental sensor inputs:

| Input | Description | Unit |
|---|---|---|
| PM2.5 | Fine particulate matter | µg/m³ |
| PM10 | Coarse particulate matter | µg/m³ |
| NO₂ | Nitrogen dioxide | ppb |
| SO₂ | Sulfur dioxide | ppb |
| CO | Carbon monoxide | ppm |
| O₃ | Ozone | ppb |
| Temperature | Ambient temperature | °C |
| Humidity | Relative humidity | % |

The model outputs a **numeric AQI score** and a **health category** (Good → Hazardous) that tells users whether the air is safe to breathe.

---

## Features

- **8 interactive input fields** — sliders + number boxes, fully synchronized
- **Live AQI gauge** — animated semicircle that fills and changes color by severity
- **Health category badge** — color-coded from green (Good) to dark red (Hazardous)
- **Pollutant breakdown bars** — visual comparison of each input's relative level
- **AQI scale reference** — built-in legend with health descriptions
- **Flask REST API** — clean `/predict` endpoint returns JSON
- **Single-terminal setup** — Flask serves both UI and API from `localhost:5000`
- **Error handling** — red banner shown if Flask API is unreachable
- **Keyboard shortcut** — press `Enter` to trigger prediction

---

## Tech Stack

**Backend**
- Python 3.10+
- Flask — web server and REST API
- scikit-learn — ML model training and prediction
- pandas / numpy — data processing and feature engineering
- joblib — model serialization (.pkl files)

**Frontend**
- HTML5 / CSS3 / Vanilla JavaScript
- Google Fonts (Syne + Space Mono)
- Fetch API for async calls to Flask

**ML**
- Algorithm: Linear Regression (best of 8 tested)
- R² Score: **0.9563**
- RMSE: **10.99 AQI units**
- Cross-Validation R²: **0.9580 ± 0.004**

---

## Project Structure

```
aqi-predictor/
│
├── backend/                           # Python Flask application
│   ├── app.py                         # Flask server + /predict API route
│   ├── model_trainer.py               # Train model and save .pkl files
│   ├── requirements.txt               # Python dependencies
│   ├── templates/
│   │   └── index.html                 # Main UI (served at localhost:5000)
│   └── model/
│       ├── linear_model.pkl           # Saved trained model
│       └── scaler.pkl                 # Saved StandardScaler
│
├── data/
│   └── air_quality_1000_dataset.csv   # Training dataset (1000 samples)
│
├── notebook/
│   └── Air_Quality_Prediction.ipynb   # Full ML pipeline notebook
│
└── README.md
```

---

## Installation & Setup

### Prerequisites

Make sure you have the following installed:

- Python 3.10 or higher — https://python.org
- pip (comes with Python)

### Step 1 — Download the Project

```bash
git clone https://github.com/yourusername/aqi-predictor.git
cd aqi-predictor
```

Or download the ZIP and extract it.

### Step 2 — Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

`requirements.txt` contents:

```
flask
flask-cors
scikit-learn
pandas
numpy
joblib
```

### Step 3 — Train the Model

Run the trainer script once to generate the `.pkl` files:

```bash
cd backend
python model_trainer.py
```

Expected output:

```
✅ Model and scaler saved to model/
```

This creates:
- `backend/model/linear_model.pkl` — trained Linear Regression model
- `backend/model/scaler.pkl` — fitted StandardScaler

> You only need to run this once. The `.pkl` files are reused every time Flask starts.

---

## How to Run

Start the Flask server from inside the `backend/` folder:

```bash
cd backend
python app.py
```

Expected output:

```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

Open your browser and visit:

```
http://localhost:5000
```

The full UI loads and is ready to use.

---

## How It Works

### Data Flow

```
User inputs values in the UI
        ↓
JavaScript collects 8 sensor values as JSON
        ↓
POST /predict  →  Flask API (app.py)
        ↓
Feature Engineering applied:
  PM_ratio       = PM2.5 / PM10
  NO2_SO2_sum    = NO2 + SO2
  Pollution_idx  = PM2.5×0.4 + PM10×0.3 + NO2×0.2 + SO2×0.1
        ↓
StandardScaler transforms 11 features
        ↓
LinearRegression.predict() returns AQI value
        ↓
AQI mapped to health category
        ↓
JSON response returned to browser
        ↓
UI updates gauge, badge, bars, and legend
```

### Feature Engineering

Three additional features are computed from raw inputs before prediction:

| Feature | Formula | Purpose |
|---|---|---|
| `PM_ratio` | PM2.5 ÷ PM10 | Fine-to-coarse particle ratio |
| `NO2_SO2_sum` | NO2 + SO2 | Combined gas pollutant load |
| `Pollution_index` | PM2.5×0.4 + PM10×0.3 + NO2×0.2 + SO2×0.1 | Weighted pollution score |

---

## ML Model Details

Eight regression algorithms were trained and compared in the Jupyter notebook:

| Model | R² | RMSE | MAE | CV R² |
|---|---|---|---|---|
| **Linear Regression ★** | **0.9563** | **10.99** | **8.79** | **0.9580** |
| Ridge Regression | 0.9562 | 10.99 | 8.80 | 0.9580 |
| Random Forest | 0.9462 | 12.19 | 9.86 | 0.9440 |
| Gradient Boosting | 0.9456 | 12.26 | 9.72 | 0.9491 |
| Extra Trees | 0.9441 | 12.42 | 9.96 | 0.9461 |
| Support Vector Regressor | 0.9294 | 13.96 | 10.78 | 0.9300 |
| Decision Tree | 0.8915 | 17.31 | 13.69 | 0.9049 |
| K-Nearest Neighbors | 0.8885 | 17.54 | 14.19 | 0.8717 |

**Linear Regression** was selected as the best model. It explains **95.63%** of AQI variance with an average prediction error of ±8.79 AQI units.

### Why Linear Regression Won

Linear Regression outperformed more complex models because AQI is fundamentally a weighted combination of pollutant concentrations — a linear relationship. The engineered `Pollution_index` feature also captured this structure explicitly, making the linear model extremely effective on this dataset.

---

## API Reference

### POST /predict

Accepts sensor values and returns the predicted AQI and health category.

**Request Body (JSON)**

```json
{
  "pm25":     120.0,
  "pm10":     200.0,
  "no2":       60.0,
  "so2":       30.0,
  "co":         1.2,
  "o3":        75.0,
  "temp":      28.0,
  "humidity":  60.0
}
```

**Response (JSON)**

```json
{
  "aqi": 137.28,
  "category": "Unhealthy for Sensitive"
}
```

**Example with curl**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"pm25":120,"pm10":200,"no2":60,"so2":30,"co":1.2,"o3":75,"temp":28,"humidity":60}'
```

---

### GET /health

Health check to verify the server is running.

**Response**

```json
{ "status": "running" }
```

---

## AQI Categories

| AQI Range | Category | Health Implication |
|---|---|---|
| 0 – 50 | Good | Satisfactory. No health risk. |
| 51 – 100 | Moderate | Acceptable. Sensitive individuals may be mildly affected. |
| 101 – 150 | Unhealthy for Sensitive | Children, elderly, and those with lung/heart conditions affected. |
| 151 – 200 | Unhealthy | Everyone may experience adverse health effects. |
| 201 – 300 | Very Unhealthy | Health alert. Everyone at significant risk. |
| 301+ | Hazardous | Emergency conditions. Entire population affected. |

---

## Dataset

| Property | Value |
|---|---|
| File | `data/air_quality_1000_dataset.csv` |
| Samples | 1,000 rows |
| Features | 8 sensor readings |
| Target | AQI (continuous, range 26–296) |
| Train split | 800 samples (80%) |
| Test split | 200 samples (20%) |
| Missing values | None |
| Outliers | None detected (IQR method) |

---

## License

This project is for educational purposes.

---

*Built with Python, Flask, scikit-learn, and a lot of coffee*
