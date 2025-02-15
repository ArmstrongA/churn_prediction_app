# churn_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging
from feature_selection import FeatureSelector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Data
df = pd.read_csv("./data/model_data.csv")

# Feature Selection

selected_features = joblib.load('boruta_features.pkl')

# Prepare Data
X = df[selected_features]
y = df["Churn Label"].apply(lambda x: 1 if x == "Yes" else 0)  # Convert target to binary

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save Scaler
joblib.dump(scaler, "churn_scaler.pkl")

# Define Model
rf = RandomForestClassifier(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best Model
best_model = grid_search.best_estimator_
joblib.dump(best_model, "churn_model.pkl")

# Evaluate Model
y_pred = best_model.predict(X_test_scaled)
logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
logger.info(f"Classification Report:\n {classification_report(y_test, y_pred)}")
