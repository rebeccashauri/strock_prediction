from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
FEATURES = [
    "gender", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status",
]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]


data = pd.read_csv(BASE_DIR / "stroke.csv").drop(columns=["id"], errors="ignore").dropna()
for column in CATEGORICAL:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column].astype(str))

X = data[FEATURES]
y = data["stroke"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
scaler.fit(X_train)

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
)
calibrated_model = CalibratedClassifierCV(
    model,
    method="sigmoid",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
)
calibrated_model.fit(scaler.transform(X_train), y_train)

# Refit the calibrated model on all available labeled data.
scaler.fit(X)
calibrated_model.fit(scaler.transform(X), y)
joblib.dump(calibrated_model, BASE_DIR / "stroke_model.pkl")
joblib.dump(scaler, BASE_DIR / "scaler.pkl")
print("Saved stroke_model.pkl and scaler.pkl using stratified calibrated training.")