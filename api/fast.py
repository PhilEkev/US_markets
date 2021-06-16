from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from tensorflow.keras import models

import yfinance as yf
import pandas as pd



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict")
def predict(company):
    
    stock = company
    
    start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    data = yf.download(tickers=stock, period='1d', start=start, end=end,)
    X_pred = data['Adj Close']
    
    model = models.load_model("notremodel")
    results = model.predict(X_pred)
    y_pred = float(results[0])
    return dict(prediction=y_pred)
