# NeuroWatch Stroke Risk Dashboard

NeuroWatch is a hospital-facing Flask dashboard for simulating stroke-risk screening from the stroke prediction dataset. It connects a calibrated XGBoost model to a responsive clinical-style workflow for entering patient observations and viewing a probability-based risk signal.

> **Clinical disclaimer:** This project is an educational decision-support demo. It is not a medical device, diagnosis, or substitute for a qualified clinician or local stroke protocol.

## Features

- Hospital-style screening dashboard for age, clinical history, vitals, residence, work, and smoking history
- Calibrated probability output with Low risk, Watch, and High risk screening bands
- Saved model and scaler artifacts for fast local startup
- Automated low-, medium-, and high-risk screening smoke tests
- `/health` endpoint for checking model availability
- Dataset `id` and the optional patient identifier are excluded from model features

## Project structure

| File or folder | Purpose |
| --- | --- |
| `app.py` | Flask application and prediction route |
| `templates/index.html` | Hospital dashboard template |
| `static/app.css` | Dashboard styling and responsive layout |
| `train_model.py` | Reproducible calibrated model training |
| `test_screening.py` | Low, medium, and high screening checks |
| `stroke.csv` | Source dataset |
| `stroke_model.pkl` | Saved calibrated classifier |
| `scaler.pkl` | Saved feature scaler |
| `Stroke prediction.ipynb` | Original exploration and modeling notebook |

## Setup

Use Python 3.10 or newer in a virtual environment when possible:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train and test the model

Retrain the saved artifacts from the CSV:

```powershell
python train_model.py
```

Run the screening checks:

```powershell
python test_screening.py
```

The test command verifies that representative profiles produce Low risk, Watch, and High risk results.

## Run the dashboard

```powershell
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser. The health endpoint is available at [http://127.0.0.1:5000/health](http://127.0.0.1:5000/health).

## Live deployment

Open the hosted application at [https://ai-based-strock-prediction.onrender.com/](https://ai-based-strock-prediction.onrender.com/).

For Render, use:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --bind 0.0.0.0:$PORT app:app`

## Model notes

The application uses ten features: gender, age, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, and smoking status. Categorical values are encoded from the source data, and the `id` column is removed before scaling and prediction. The training script uses a stratified split and cross-validated sigmoid calibration so the displayed probability is more useful for screening than the default uncalibrated classifier score.