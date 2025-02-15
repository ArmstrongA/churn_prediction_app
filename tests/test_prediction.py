# tests/test_prediction.py
import pytest
import pandas as pd
import numpy as np
from prediction_pipeline import predict_churn

# def test_predict_churn_single():
#     test_data = {
#         "Gender": "Female",
#         "Senior Citizen": "No",
#         "Partner": "Yes",
#         "Dependents": "No",
#         "Tenure Months": 24,
#         "Phone Service": "Yes",
#         "Multiple Lines": "No",
#         "Internet Service": "DSL",
#         "Online Security": "Yes",
#         "Online Backup": "No",
#         "Device Protection": "Yes",
#         "Tech Support": "No",
#         "Streaming TV": "Yes",
#         "Streaming Movies": "No",
#         "Contract": "Month-to-month",
#         "Paperless Billing": "Yes",
#         "Payment Method": "Electronic check",
#         "Monthly Charges": 65.6,
#         "Total Charges": 1576.45,
#         "customer_text": "Your internet is horrible these days!"
#     }
    
#     result = predict_churn(test_data, "churn_model.pkl", "scaler.pkl")
#     assert isinstance(result, dict)
#     assert 'churn_prediction' in result
#     assert 'churn_probability' in result
#     assert isinstance(result['churn_prediction'], list)
#     assert isinstance(result['churn_probability'], list)

def test_predict_churn_batch():
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
            "customer_text": "Your internet is horrible these days!"
        },
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
            "customer_text": "Your internet is horrible these days!"
        }
    ]
    
    result = predict_churn(test_data, "churn_model.pkl", "churn_scaler.pkl")
    assert isinstance(result, dict)
    assert 'churn_prediction' in result
    assert 'churn_probability' in result
    assert isinstance(result['churn_prediction'], list)
    assert isinstance(result['churn_probability'], list)
    assert len(result['churn_prediction']) == 1
    assert len(result['churn_probability']) == 1