# FastAPI app (main.py or app.py)
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Union
# Import predict_churn 
from prediction_pipeline import predict_churn
import os
app = FastAPI()

# Define input data model (Pydantic) for validation
class CustomerData(BaseModel):
    # Define your expected features here:

    Gender: str = None  # Make all fields optional, or provide default values
    SeniorCitizen: str = Field(None, alias="Senior Citizen")
    Partner: str = None
    Dependents: str = None
    TenureMonths: float = Field(None, alias="Tenure Months")
    PhoneService: str = Field(None, alias="Phone Service")
    MultipleLines: str = Field(None, alias="Multiple Lines")
    InternetService: str = Field(None, alias="Internet Service")
    OnlineSecurity: str = Field(None, alias="Online Security")
    OnlineBackup: str = Field(None, alias="Online Backup")
    DeviceProtection: str = Field(None, alias="Device Protection")
    TechSupport: str = Field(None, alias="Tech Support")
    StreamingTV: str = Field(None, alias="Streaming TV")
    StreamingMovies: str = Field(None, alias="Streaming Movies")
    Contract: str = None
    PaperlessBilling: str = Field(None, alias="Paperless Billing")
    PaymentMethod: str = Field(None, alias="Payment Method")
    MonthlyCharges: float = Field(None, alias="Monthly Charges")
    TotalCharges: str = Field(None, alias="Total Charges")
    customer_text: str = None
class Config:
        allow_population_by_field_name = True
        
@app.post("/predict")
async def predict_endpoint(request: Request, data: Union[CustomerData, List[CustomerData], Dict, List[Dict]]): 
    print(data)  # Print the data
    import logging
    logging.info(f"Received data: {data}") # Log the data
    # Accept single or batch
    """Endpoint for making churn predictions."""

    model_path = "churn_model.pkl"
    scaler_path = "churn_scaler.pkl"

    try:
        if isinstance(data, CustomerData):  # Single prediction
            data = data.model_dump()
        elif isinstance(data, list) and all(isinstance(item, CustomerData) for item in data): # Batch prediction of validated data
            data = [item.model_dump() for item in data]
        elif isinstance(data, dict): # Single prediction with a dictionary
            data = data
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data): # Batch prediction with list of dictionaries
            data = data
        else:
            raise ValueError("Invalid data format. Please provide a dictionary or a list of dictionaries or Pydantic model instances.")

        predictions = predict_churn(data, model_path, scaler_path)
        return predictions

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))  # Bad Request
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")  # Internal Server Error