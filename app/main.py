from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

model_path = os.path.join(os.path.dirname(__file__), '../model.pkl')
model = joblib.load(model_path)

class PredictionRequest(BaseModel):
    data: list

@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([request.data])
    return {"prediction": prediction.tolist()}
