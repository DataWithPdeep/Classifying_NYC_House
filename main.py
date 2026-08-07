from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field
import joblib
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COLUMNS = ["latitude", "longitude", "price", "minimum_nights",
    "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
    "neighbourhood_group", "neighbourhood",]

model = joblib.load("Model_Pipeline.pkl")



class Features(BaseModel):
    latitude: float = Field(..., get=-90, le=90, description="Latitude must be between -90 and 90")
    longitude: float = Field(..., get=-180, le=180, description="Longitude must be between -180 and 180")
    price: float = Field(..., ge=0, description="Price must be a positive number")
    minimum_nights: int = Field(..., ge=0, description="Minimum nights must be a non-negative integer")
    number_of_reviews: int = Field(..., ge=0, description="Number of reviews must be a non-negative integer")
    reviews_per_month: float = Field(..., ge=0, description="Reviews per month must be a non-negative number")
    calculated_host_listings_count: int = Field(..., ge=0, description="Calculated host listings count must be a non-negative integer")
    availability_365: int = Field(..., ge=0, le=365, description="Availability must be an integer between 0 and 365")
    neighbourhood_group: str = Field(..., description="Neighbourhood group must be a string")
    neighbourhood: str = Field(..., description="Neighbourhood must be a string")

@app.get("/")
def greet():
    return "Hello Guyss"


@app.post("/predict")
def predict(features):
    row = pd.DataFrame(features.dict(), columns=COLUMNS)
    prediction = model.predict(row)
    probability = model.predict_proba(row)

    return {
            "Predicted_room_type": prediction,
            "Probability": probability.tolist()[0]
            }


    