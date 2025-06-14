from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import src.predict as pr

app = FastAPI(title="Fraud-Detection API")

class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float
    V7: float; V8: float; V9: float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float; V17: float
    V18: float; V19: float; V20: float; V21: float; V22: float
    V23: float; V24: float; V25: float; V26: float; V27: float; V28: float

@app.post("/predict")
def predict(tx: Transaction):
    try:
        pred = pr.predict_one(tx.dict())
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))
    return {"prediction": pred}
