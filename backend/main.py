from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
import io

app = FastAPI()

#CORS dovoljenja
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    with open("disk_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("OPOZORILO: model.pkl ni najden. API bo deloval v testnem načinu.")

#glavni api ki prejme moj model
@app.post("/api/analyze-scan")
async def analyze_scan(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Samo .csv datoteke so dovoljene.")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # 2. Tukaj bi običajno filtrirali dataframe, da obdržiš samo featurese za model
    # npr. features = df[['smart_5_raw', 'smart_187_raw', 'age_days']]

    if model:
        # 3. Model Inference
        # predictions = model.predict(features)

        # Simulacija rezultata iz modela za ta primer:
        risk_score = 0.98
        verdict = "Critical"
    else:
        # Fallback, če modela še ni
        risk_score = 0.98
        verdict = "Critical"

    # 4. Vrnemo strukturiran JSON nazaj v React
    return {
        "status": "success",
        "filename": file.filename,
        "results": {
            "risk_score": risk_score,
            "verdict": verdict,
            "anomalies_detected": len(df),
            "critical_features": {
                "smart_5_raw": 144,
                "smart_187_raw": 23
            }
        }
    }

# Zaženi s komando: uvicorn main:app --reload