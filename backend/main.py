import json
import os
import sys
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

SRCML_PATH = "/app/srcML"
if SRCML_PATH not in sys.path:
    sys.path.insert(0, SRCML_PATH)

from srcML.disk_pipeline import pretvori_json_v_surovi_df
app = FastAPI(title="TrueNAS Smart Scan Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PIPELINE_PATH = '/app/srcML/disk_health_pipeline.pkl'

try:
    pipeline = joblib.load(PIPELINE_PATH)
    print("ML Pipeline uspešno naložen v spomin!")
except Exception as e:
    print(f"Napaka pri nalaganju pkl datoteke: {e}")
    pipeline = None


@app.post("/api/analyze-smart-json")
async def analyze_smart_json(file: UploadFile = File(...)):
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Naložiti morate veljavno JSON datoteko.")

    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model strojnega učenja ni na voljo na strežniku.")

    try:
        surova_vsebina = await file.read()
        smartctl_dict = json.loads(surova_vsebina)

        surovi_df = pretvori_json_v_surovi_df(smartctl_dict)
        analiza_rezultat = pipeline.analyze(surovi_df)

        return analiza_rezultat
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Napaka med analizo podatkov: {str(e)}")