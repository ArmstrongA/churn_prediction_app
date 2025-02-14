# feature_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
import math
import os

class FeatureAnalysis:
    def __init__(self, df):
        self.df = df
        self.numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        self.categorical_cols = df.select_dtypes(include=['object']).columns
        
    def plot_feature_distributions(self, save_path=None):
        """Plot distributions of numerical features."""
        num_features = len(self.numerical_cols)
        num_cols = 4  # Set columns to 4 for better spacing
        num_rows = math.ceil(num_features / num_cols)  # Adjust rows dynamically
        
        plt.figure(figsize=(15, 5 * num_rows))  # Adjust figure size based on rows
        for i, col in enumerate(self.numerical_cols, 1):
            plt.subplot(num_rows, num_cols, i)
            sns.histplot(data=self.df, x=col, hue='Churn Label', alpha=0.5)
            plt.title(f'{col} Distribution')
            plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'feature_distributions.png'))

        plt.show()

    def plot_correlation_matrix(self, save_path=None):
        """Plot correlation matrix of numerical features."""
        corr_matrix = self.df[self.numerical_cols].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'correlation_matrix.png'))

        plt.show()
        
    def plot_feature_importance(self, target_col='Churn Label', save_path=None):
        """Calculate and plot feature importance using mutual information."""
        X = self.df[self.numerical_cols]
        y = (self.df[target_col] == 'Yes').astype(int)
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X, y)
        importance_df = pd.DataFrame({
            'Feature': self.numerical_cols,
            'Importance': mi_scores
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.title('Feature Importance (Mutual Information)')

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'feature_importance.png'))
        
        return importance_df
    
    def plot_categorical_analysis(self, save_path=None):
        """Analyze categorical features relationship with churn."""
        num_features = len(self.categorical_cols)
        num_cols = 3  # Set to 3 columns for a cleaner view
        num_rows = math.ceil(num_features / num_cols)  # Dynamically adjust rows

        plt.figure(figsize=(15, 5 * num_rows))
        for i, col in enumerate(self.categorical_cols, 1):
            if col != 'Churn Label':
                plt.subplot(num_rows, num_cols, i)
                churn_props = self.df.groupby(col)['Churn Label'].value_counts(normalize=True).unstack()
                if 'Yes' in churn_props.columns:
                    churn_props['Yes'].sort_values().plot(kind='bar')
                else:
                    churn_props.plot(kind='bar', stacked=True)
                plt.title(f'Churn Rate by {col}')
                plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'categorical_analysis.png'))
        plt.show()
    
    def generate_feature_report(self, save_path=None):
        """Generate a comprehensive feature analysis report."""
        report = {
            'numerical_stats': self.df[self.numerical_cols].describe(),
            'categorical_stats': {
                col: self.df[col].value_counts(normalize=True)
                for col in self.categorical_cols
            },
            'missing_values': self.df.isnull().sum(),
            'correlation_analysis': self.df[self.numerical_cols].corr()
        }
        
        if save_path:
            with open(f'{save_path}/feature_report.txt', 'w') as f:
                for section, data in report.items():
                    f.write(f'\n{section.upper()}\n{"="*50}\n')
                    f.write(str(data))
                    f.write('\n\n')
                    
        return report

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv('processed_telco_data.csv')
    df = df.drop(columns='conversation')
    
    # Create analysis object
    analyzer = FeatureAnalysis(df)
    
    # Generate visualizations
    analyzer.plot_feature_distributions('outputs')
    analyzer.plot_correlation_matrix('outputs')
    analyzer.plot_feature_importance(target_col='Churn Label', save_path='outputs')
    analyzer.plot_categorical_analysis('outputs')
    
    # Generate report
    report = analyzer.generate_feature_report('outputs')
