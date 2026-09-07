🏠 NYC Airbnb Room Type Classification

A machine learning classification project that predicts the room type
of an NYC Airbnb listing --- Entire home/apt, Private room, or
Shared room --- using listing characteristics such as location, price,
minimum stay, review activity, host listing count, and availability.

The trained model is exposed through a FastAPI REST API and a custom
HTML/CSS/JavaScript frontend for interactive predictions.

🚀 Project Overview

This project uses the NYC Airbnb Open Data dataset to build a supervised
machine learning classification pipeline.

Given information about an Airbnb listing, the application predicts the
most likely room type and returns the prediction probabilities for each
class.

Prediction Classes

🏠 Entire home/apt

🚪 Private room

🛏️ Shared room

✨ Features

Machine learning classification for NYC Airbnb room types

Scikit-learn model pipeline saved with Joblib

FastAPI backend for real-time inference

Pydantic input validation

Interactive web interface

Prediction probabilities for all room-type classes

API health-check endpoint

Example listing button in the frontend

Responsive custom HTML/CSS/JavaScript UI

CORS enabled for API access

🧠 Input Features

The prediction API accepts the following features:

Feature                             Description

latitude                          Geographic latitude of the listing

longitude                         Geographic longitude of the listing

price                             Price per night

minimum_nights                    Minimum number of nights required

number_of_reviews                 Total number of reviews

reviews_per_month                 Average reviews received per month

calculated_host_listings_count    Number of listings managed by the
host

availability_365                  Number of days available during a
year

neighbourhood_group               NYC borough

neighbourhood                     NYC neighbourhood

The FastAPI layer validates numerical ranges such as latitude/longitude,
non-negative prices and review counts, and availability between 0 and
365 days.

🔄 Prediction Workflow

NYC Airbnb Listing
        │
        ▼
User enters listing details
        │
        ▼
HTML / CSS / JavaScript Frontend
        │
        ▼
POST /predict
        │
        ▼
Pydantic Validation
        │
        ▼
Pandas DataFrame
        │
        ▼
Scikit-learn Model Pipeline
        │
        ▼
Room Type + Class Probabilities
        │
        ▼
Interactive Prediction UI

🏗️ Architecture

┌──────────────────────────────┐
│       Web Frontend           │
│ HTML + CSS + JavaScript      │
└──────────────┬───────────────┘
               │
               │ HTTP POST
               ▼
┌──────────────────────────────┐
│        FastAPI API           │
│                              │
│  Pydantic Validation         │
│  /predict                    │
│  /api                        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Scikit-learn Model         │
│   Model_Pipeline.pkl         │
└──────────────┬───────────────┘
               │
               ▼
     Prediction + Probability

📊 Dataset

The project notebook downloads the New York City Airbnb Open Data
dataset through kagglehub and loads AB_NYC_2019.csv.

Dataset reference used in the notebook:

dgomonov/new-york-city-airbnb-open-data

The notebook begins the data science workflow with NumPy, Pandas,
Matplotlib and Seaborn, followed by model development and pipeline
creation.

🤖 Model

The trained model is stored as:

Model_Pipeline.pkl

The FastAPI application loads this pipeline using Joblib and calls:

model.predict(row)
model.predict_proba(row)

The API therefore returns both the predicted room type and the
probability distribution produced by the trained classifier.

🌐 API Endpoints

GET /

Serves the interactive frontend.

GET /api

Health-check endpoint.

Example response:

{
  "message": "NYC House Classification API is Running Successfully 🚀"
}

POST /predict

Predicts the room type for a listing.

Example Request

{
  "latitude": 40.7484,
  "longitude": -73.9857,
  "price": 120,
  "minimum_nights": 2,
  "number_of_reviews": 84,
  "reviews_per_month": 2.3,
  "calculated_host_listings_count": 1,
  "availability_365": 210,
  "neighbourhood_group": "Manhattan",
  "neighbourhood": "Midtown"
}

Example Response

{
  "Predicted_room_type": "Entire home/apt",
  "Probability": [
    0.82,
    0.15,
    0.03
  ]
}

The exact prediction and probabilities depend on the trained model and
input data.

🖥️ Frontend

The project includes a custom frontend built with:

HTML

CSS

Vanilla JavaScript

The interface allows users to enter:

Latitude and longitude

NYC borough and neighbourhood

Price per night

Minimum nights

Availability

Number of reviews

Reviews per month

Host listing count

After prediction, the UI displays the most likely room type and visual
probability comparisons for the available classes.

📁 Project Structure

Classifying_NYC_House/
│
├── Model_Pipeline.pkl       # Trained ML pipeline
├── nyc.ipynb                # Data analysis & ML notebook
│
├── main.py                  # FastAPI application
├── index.html               # Frontend
├── script.js                # Frontend logic
├── style.css                # Frontend styling
│
├── requirements.txt         # Python dependencies
├── runtime.txt              # Runtime configuration
├── .python-version          # Python version configuration
└── .gitattributes

🛠️ Tech Stack

Programming & Data Science

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Joblib

Backend

FastAPI

Uvicorn

Pydantic

Frontend

HTML

CSS

JavaScript

Deployment / Runtime

FastAPI-compatible ASGI server

Uvicorn

📦 Requirements

The project pins the following main dependencies:

fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
pandas==2.2.3
scikit-learn==1.6.1
joblib==1.4.2

⚙️ Installation

1. Clone the repository

git clone https://github.com/DataWithPdeep/Classifying_NYC_House.git
cd Classifying_NYC_House

2. Create a virtual environment

python -m venv venv

3. Activate the environment

Windows

venv\Scripts\activate

macOS / Linux

source venv/bin/activate

4. Install dependencies

pip install -r requirements.txt

▶️ Run the Application

Start the FastAPI server:

uvicorn main:app --reload

The application will be available at:

http://127.0.0.1:8000

Open the URL in your browser to use the interactive prediction
interface.

📚 API Documentation

FastAPI automatically provides interactive API documentation.

Swagger UI

http://127.0.0.1:8000/docs

ReDoc

http://127.0.0.1:8000/redoc

🧪 Testing the Prediction API

You can test the endpoint using curl:

curl -X POST "http://127.0.0.1:8000/predict" ^
-H "Content-Type: application/json" ^
-d "{\"latitude\":40.7484,\"longitude\":-73.9857,\"price\":120,\"minimum_nights\":2,\"number_of_reviews\":84,\"reviews_per_month\":2.3,\"calculated_host_listings_count\":1,\"availability_365\":210,\"neighbourhood_group\":\"Manhattan\",\"neighbourhood\":\"Midtown\"}"

On macOS/Linux, use the equivalent JSON payload with the shell's
standard quoting.

🔍 Validation

The API uses Pydantic to validate incoming data.

Examples of validation rules include:

Latitude: -90 to 90

Longitude: -180 to 180

Price: non-negative

Minimum nights: non-negative

Number of reviews: non-negative

Reviews per month: non-negative

Host listing count: non-negative

Availability: 0 to 365

Borough and neighbourhood: string values

Invalid inputs are rejected before being passed to the machine learning
model.

🎯 Project Objective

The goal of this project is to demonstrate an end-to-end machine
learning workflow:

Obtain a real-world dataset

Explore and prepare the data

Build a classification model

Save the trained pipeline

Create an inference API

Validate incoming requests

Build a user-friendly frontend

Return real-time predictions and probabilities

🔮 Future Improvements

Potential extensions include:

Add model performance metrics to the README

Add confusion matrix and classification report

Add automated model evaluation

Add Docker support

Add CI/CD with GitHub Actions

Add model versioning

Add automated API tests

Add logging and monitoring

Deploy the frontend and API independently

Add more advanced feature engineering

⚠️ Disclaimer

This project is intended for educational and portfolio purposes.
Predictions are based on a trained machine learning model and should not
be treated as definitive information about Airbnb listings.

🔗 Repository

GitHub:

https://github.com/DataWithPdeep/Classifying_NYC_House

👨‍💻 Author

Pradeep Singh

GitHub: https://github.com/DataWithPdeep

Kaggle: https://www.kaggle.com/psrana344

⭐ If you find this project useful, consider giving the repository a
star.
