from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict

# Initialize FastAPI
app = FastAPI(
    title="ChurnGuard ML API",
    description="Customer churn prediction API using XGBoost",
    version="1.0.0"
)

# Load model and artifacts
try:
    model = joblib.load('../models/churnguard_model.joblib')
    label_encoders = joblib.load('../models/label_encoders.joblib')
    feature_names = joblib.load('../models/feature_names.joblib')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Define input schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 840.0
            }
        }

@app.get("/")
def root():
    return {
        "message": "ChurnGuard ML API",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_churn(customer: CustomerData) -> Dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dataframe
        df = pd.DataFrame([customer.dict()])
        
        # Add engineered feature
        df['customer_value'] = df['MonthlyCharges'] * df['tenure']
        
        # Encode categorical variables
        for col in label_encoders.keys():
            if col in df.columns:
                try:
                    df[col] = label_encoders[col].transform(df[col])
                except ValueError:
                    # Handle unseen categories
                    df[col] = 0
        
        # Ensure correct feature order
        df = df[feature_names]
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        
        # Interpret results
        churn_prob = float(probability[1])
        risk_level = "High" if churn_prob > 0.7 else "Medium" if churn_prob > 0.4 else "Low"
        
        return {
            "prediction": "Will Churn" if prediction == 1 else "Will Not Churn",
            "churn_probability": round(churn_prob, 4),
            "retention_probability": round(1 - churn_prob, 4),
            "risk_level": risk_level,
            "confidence": "High" if max(probability) > 0.8 else "Medium" if max(probability) > 0.6 else "Low"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)