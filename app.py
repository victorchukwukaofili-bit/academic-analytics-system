import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Academic Early Warning System API",
    description="Inference microservice for predicting student academic risk.",
    version="1.0.0"
)

# Load saved model on startup
model = joblib.load("model.pkl")

class StudentFeatures(BaseModel):
    entry_mode_code: int
    socio_band_code: int
    prior_cgpa: float
    ca_mean: float
    ca_variance: float
    momentum: float
    attendance_rate: float
    clicks_total: int
    duration_total_min: float
    resource_breadth: int
    submission_timeliness: float
    engagement_index: float
    forum_posts_count: int
    login_frequency: int

@app.get("/")
def health_check():
    return {"status": "online", "system": "Academic Early Warning System API"}

@app.post("/predict")
def predict_risk(student: StudentFeatures):
    input_data = pd.DataFrame([student.dict()])
    prob_risk = float(model.predict_proba(input_data)[0][1])
    is_at_risk = bool(prob_risk >= 0.5)
    
    risk_level = "High" if prob_risk > 0.7 else ("Medium" if prob_risk > 0.4 else "Low")
    
    return {
        "at_risk": is_at_risk,
        "risk_probability": round(prob_risk, 4),
        "risk_level": risk_level
    }