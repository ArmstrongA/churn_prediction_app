import pandas as pd
import joblib

from data_preparation_ import DataPreparation

def predict_churn(data, model_path, scaler_path):
    try:
        # 1. Convert data to DataFrame (handle various input types)
        import pandas as pd
        if isinstance(data, pd.DataFrame):
                new_df = data
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                new_df = pd.DataFrame(data)
        elif isinstance(data, dict):  # Single prediction 
                new_df = pd.DataFrame([data], index=[0]) 
        else:
            raise ValueError("Invalid data format. Provide a DataFrame, list of dicts, or a single dict.")

        # 2. Preprocess Data
        prep = DataPreparation()
        new_df, validation_report = prep.prepare_data(new_df)

        # 3. Feature Selection
        selected_features = joblib.load('boruta_features.pkl')
        X_new = new_df[selected_features]

        # 4. Handle Missing Columns
        missing_cols = set(X_new.columns) - set(selected_features)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # 5. Scale Data
        scaler = joblib.load(scaler_path)
        X_new_scaled = scaler.transform(X_new)

        # 6. Load Model
        model = joblib.load(model_path)

        # 7. Make Predictions
        y_prob = model.predict_proba(X_new_scaled)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # 8. Create Response
        predictions = {
            "churn_probability": y_prob.tolist(),
            "churn_prediction": y_pred.tolist()
        }
        return predictions

    except FileNotFoundError:
        return {"error": "Model or scaler file not found."}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}