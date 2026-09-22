NEUROWATCH STROKE RISK DASHBOARD
================================

Live application:
https://ai-based-strock-prediction.onrender.com/

NeuroWatch is a hospital-facing Flask dashboard for simulating stroke-risk
screening from a stroke prediction dataset. It uses a calibrated XGBoost model
to display Low risk, Watch, and High risk screening signals.

IMPORTANT:
This is an educational decision-support demo. It is not a medical device,
diagnosis, or substitute for a qualified clinician or local stroke protocol.

RENDER DEPLOYMENT
-----------------
Build command:
pip install -r requirements.txt

Start command:
gunicorn --bind 0.0.0.0:$PORT app:app

LOCAL SETUP
-----------
1. Create and activate a virtual environment:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2. Install dependencies:
   python -m pip install -r requirements.txt

3. Run screening tests:
   python test_screening.py

4. Start the local dashboard:
   python app.py

Local URL:
http://127.0.0.1:5000

Health check:
http://127.0.0.1:5000/health

MODEL FEATURES
--------------
The model uses ten features: gender, age, hypertension, heart disease,
marital status, work type, residence type, average glucose level, BMI, and
smoking status. The dataset id column and patient identifiers are not used as
model features.