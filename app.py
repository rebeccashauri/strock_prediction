from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
MODEL = joblib.load(BASE_DIR / "stroke_model.pkl")
SCALER = joblib.load(BASE_DIR / "scaler.pkl")

FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]
ENCODED_COLUMNS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]


def build_encoders():
    """Recreate the notebook's independent encoder for each categorical field."""
    data = pd.read_csv(BASE_DIR / "stroke.csv").drop(columns=["id"], errors="ignore").dropna()
    encoders = {}
    for column in ENCODED_COLUMNS:
        encoder = LabelEncoder()
        encoder.fit(data[column].astype(str))
        encoders[column] = encoder
    return encoders


ENCODERS = build_encoders()
app = Flask(__name__)
app.config["SECRET_KEY"] = "neurowatch-local-session-key"


def form_options():
    return {
        "gender": ["Female", "Male", "Other"],
        "ever_married": ["Yes", "No"],
        "work_type": ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        "Residence_type": ["Urban", "Rural"],
        "smoking_status": ["never smoked", "formerly smoked", "smokes", "Unknown"],
    }


def encode_patient(form):
    patient = {
        "gender": form["gender"],
        "age": float(form["age"]),
        "hypertension": int(form["hypertension"]),
        "heart_disease": int(form["heart_disease"]),
        "ever_married": form["ever_married"],
        "work_type": form["work_type"],
        "Residence_type": form["Residence_type"],
        "avg_glucose_level": float(form["avg_glucose_level"]),
        "bmi": float(form["bmi"]),
        "smoking_status": form["smoking_status"],
    }
    for column, encoder in ENCODERS.items():
        if patient[column] not in encoder.classes_:
            raise ValueError(f"Unsupported value for {column}")
        patient[column] = int(encoder.transform([patient[column]])[0])
    # Keep the model boundary explicit: patient identifiers are never features.
    return pd.DataFrame([patient], columns=FEATURES)[FEATURES]


def risk_profile(probability):
    score = round(float(probability) * 100, 1)
    # Screening favors sensitivity: the model's default 0.5 class cutoff
    # would hide meaningful risk signals in this imbalanced dataset.
    if score >= 20:
        return "High risk", "urgent", "Escalate for clinician review now"
    if score >= 5:
        return "Watch", "watch", "Arrange a timely clinical review"
    return "Low risk", "low", "Continue routine prevention guidance"


@app.route("/", methods=["GET", "POST"])
def dashboard():
    assessment = session.pop("assessment", None)
    result = assessment["result"] if assessment else None
    error = None
    form_data = assessment["form_data"] if assessment else {}
    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            patient = encode_patient(request.form)
            scaled_patient = SCALER.transform(patient)
            probability = MODEL.predict_proba(scaled_patient)[0][1]
            label, tone, recommendation = risk_profile(probability)
            result = {
                "score": round(float(probability) * 100, 1),
                "label": label,
                "tone": tone,
                "recommendation": recommendation,
                "department": request.form.get("department", "Emergency medicine"),
            }
            session["assessment"] = {"result": result, "form_data": form_data}
            return redirect(url_for("dashboard"))
        except (KeyError, TypeError, ValueError) as exc:
            error = f"Please check the clinical fields and try again. ({exc})"
    return render_template("index.html", result=result, error=error, options=form_options(), form_data=form_data)


@app.get("/health")
def health():
    return {"status": "ok", "model": type(MODEL).__name__, "features": len(FEATURES)}


if __name__ == "__main__":
    app.run(debug=False, port=5000)