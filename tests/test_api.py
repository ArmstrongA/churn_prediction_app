# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

def test_predict_endpoint_single():
    test_data = {
        "Gender": "Female",
        "Senior Citizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure Months": 24,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "DSL",
        "Online Security": "Yes",
        "Online Backup": "No",
        "Device Protection": "Yes",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 65.6,
        "Total Charges": 1576.45,
        "customer_text": "You internet is horrible these days!"  
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    result = response.json()
    assert 'churn_prediction' in result
    assert 'churn_probability' in result
    assert isinstance(result['churn_prediction'], list)
    assert isinstance(result['churn_probability'], list)
    assert len(result['churn_prediction']) == 1
    assert len(result['churn_probability']) == 1

def test_predict_endpoint_batch():
    test_data = [
        {
            "Gender": "Female",
            "Senior Citizen": "No",
            "Partner": "Yes",
            "Dependents": "No",
            "Tenure Months": 24,
            "Phone Service": "Yes",
            "Multiple Lines": "No",
            "Internet Service": "DSL",
            "Online Security": "Yes",
            "Online Backup": "No",
            "Device Protection": "Yes",
            "Tech Support": "No",
            "Streaming TV": "Yes",
            "Streaming Movies": "No",
            "Contract": "Month-to-month",
            "Paperless Billing": "Yes",
            "Payment Method": "Electronic check",
            "Monthly Charges": 65.6,
            "Total Charges": 1576.45,
            "customer_text": "You internet is horrible these days!"
        },
        {
            "Gender": "Male",
            "Senior Citizen": "Yes",
            "Partner": "No",
            "Dependents": "No",
            "Tenure Months": 12,
            "Phone Service": "Yes",
            "Multiple Lines": "Yes",
            "Internet Service": "Fiber optic",
            "Online Security": "No",
            "Online Backup": "Yes",
            "Device Protection": "No",
            "Tech Support": "No",
            "Streaming TV": "Yes",
            "Streaming Movies": "Yes",
            "Contract": "Month-to-month",
            "Paperless Billing": "Yes",
            "Payment Method": "Credit card",
            "Monthly Charges": 89.1,
            "Total Charges": 1069.2,
            "customer_text": "You internet is always the best!"
        }
    ]
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    result = response.json()
    assert 'churn_prediction' in result
    assert 'churn_probability' in result
    assert isinstance(result['churn_prediction'], list)
    assert isinstance(result['churn_probability'], list)
    assert len(result['churn_prediction']) == 2
    assert len(result['churn_probability']) == 2

def test_invalid_data_missing_fields():
    test_data = {
        "Gender": "Male"
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200  # Since fields are optional
    result = response.json()
    # Check if we get either a valid prediction or an error message
    assert any([
        all(key in result for key in ['churn_prediction', 'churn_probability']),
        'error' in result
    ])