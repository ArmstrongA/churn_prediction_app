# feature_selection.py
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from boruta import BorutaPy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureSelector:
    def __init__(self, df, target_col='Churn Label'):
        self.df = df.copy()
        self.target_col = target_col
        self.le = LabelEncoder()
        
        # Ensure only numerical columns are used
        self.numerical_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        if self.target_col in self.numerical_cols:
            self.numerical_cols.remove(self.target_col)
    
    def prepare_target(self):
        """Prepare target variable for selection methods."""
        return self.le.fit_transform(self.df[self.target_col])
    
    def filter_method(self, k=10):
        """Select features using mutual information."""
        y = self.prepare_target()
        X = self.df[self.numerical_cols]
        
        # Select features
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[selector.get_support()].tolist()
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        
        logger.info(f"Selected {len(selected_features)} features using mutual information")
        return selected_features, feature_scores
    
    def wrapper_method(self, n_features=10):
        """Select features using Recursive Feature Elimination."""
        y = self.prepare_target()
        X = self.df[self.numerical_cols]
        
        # Initialize estimator
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        
        # Fit selector
        selector.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[selector.support_].tolist()
        feature_ranks = pd.DataFrame({
            'Feature': X.columns,
            'Rank': selector.ranking_
        }).sort_values('Rank')
        
        logger.info(f"Selected {len(selected_features)} features using RFE")
        return selected_features, feature_ranks
    
    def boruta_selection(self):
        """Select features using Boruta algorithm."""
        y = self.prepare_target()
        X = self.df[self.numerical_cols]
        
        # Initialize Random Forest classifier
        rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=42)
        
        # Initialize Boruta
        boruta = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
        
        # Fit Boruta
        boruta.fit(X.values, y)
        
        # Get selected features
        selected_features = X.columns[boruta.support_].tolist()
        feature_ranks = pd.DataFrame({
            'Feature': X.columns,
            'Boruta_Ranking': boruta.ranking_
        }).sort_values('Boruta_Ranking')
        
        logger.info(f"Selected {len(selected_features)} features using Boruta")
        return selected_features, feature_ranks

if __name__ == "__main__":
    df = pd.read_csv('processed_telco_data.csv')
    selector = FeatureSelector(df)
    
    selected_kbest, kbest_scores = selector.filter_method(k=10)
    selected_rfe, rfe_ranks = selector.wrapper_method(n_features=10)
    selected_boruta, boruta_ranks = selector.boruta_selection()
    
    print("Top Features from SelectKBest:", selected_kbest)
    print("Top Features from RFE:", selected_rfe)
    print("Top Features from Boruta:", selected_boruta)
