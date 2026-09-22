from app import MODEL, SCALER, encode_patient, risk_profile


CASES = {
    "low": {
        "gender": "Female", "age": "25", "hypertension": "0", "heart_disease": "0",
        "ever_married": "No", "work_type": "Private", "Residence_type": "Urban",
        "avg_glucose_level": "80", "bmi": "22", "smoking_status": "never smoked",
    },
    "medium": {
        "gender": "Male", "age": "55", "hypertension": "0", "heart_disease": "0",
        "ever_married": "Yes", "work_type": "Private", "Residence_type": "Rural",
        "avg_glucose_level": "160", "bmi": "32", "smoking_status": "formerly smoked",
    },
    "high": {
        "gender": "Male", "age": "82", "hypertension": "1", "heart_disease": "1",
        "ever_married": "Yes", "work_type": "Private", "Residence_type": "Rural",
        "avg_glucose_level": "275", "bmi": "42", "smoking_status": "smokes",
    },
}


expected = {"low": "Low risk", "medium": "Watch", "high": "High risk"}
for name, values in CASES.items():
    probability = float(MODEL.predict_proba(SCALER.transform(encode_patient(values)))[0, 1])
    label = risk_profile(probability)[0]
    print(f"{name}: {probability * 100:.1f}% -> {label}")
    assert label == expected[name], f"Expected {expected[name]}, got {label}"

print("All screening tiers passed.")