from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(title="NYC House Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COLUMNS = [
    "latitude",
    "longitude",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "neighbourhood_group",
    "neighbourhood",
]

model = None


@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("Model_Pipeline.pkl")


class Features(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    price: float = Field(..., ge=0)
    minimum_nights: int = Field(..., ge=0)
    number_of_reviews: int = Field(..., ge=0)
    reviews_per_month: float = Field(..., ge=0)
    calculated_host_listings_count: int = Field(..., ge=0)
    availability_365: int = Field(..., ge=0, le=365)
    neighbourhood_group: str
    neighbourhood: str


@app.get("/")
def home():
    return {"message": "NYC House Classification API is Running Successfully 🚀"}


@app.post("/predict")
def predict(features: Features):
    try:
        row = pd.DataFrame([features.model_dump()], columns=COLUMNS)

        prediction = model.predict(row)[0]
        probability = model.predict_proba(row)[0].tolist()

        return {
            "Predicted_room_type": prediction,
            "Probability": probability
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
