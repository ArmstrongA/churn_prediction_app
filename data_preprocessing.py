# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparation:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer()
        self.imputer = SimpleImputer(strategy='median')
        
    def load_data(self, file_path):
        """Load and validate the dataset."""
        try:
            df = pd.read_csv(file_path)
            
            logger.info(f"Successfully loaded dataset with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def validate_data(self, df):
        """Perform basic data validation checks."""
        validation_report = {
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Check for invalid values in important columns
        if 'Monthly Charges' in df.columns:
            validation_report['negative_charges'] = (df['Monthly Charges'] < 0).sum()
            
        if 'Tenure Months' in df.columns:
            validation_report['invalid_tenure'] = (df['Tenure Months'] < 0).sum()
            
        logger.info("Data validation completed")
        return validation_report
    
    def engineer_basic_features(self, df):
        """Create basic derived features."""
        df = df.copy()
        # drop unecessary columns
        df = df.drop(['CustomerID',	'Count', 'Country',	'State', 'City', 'Zip Code', 'Lat Long', 'Latitude', 
                      'Longitude', 'Churn Value', 'Churn Score',	'CLTV',	'Churn Reason', 'conversation', 'customer_text'], axis=1)

        
        # Customer value features
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
        df['Revenue_per_Month'] = df['Total Charges'] / df['Tenure Months']
        df['Average_Monthly_Charges'] = df['Total Charges'] / df['Tenure Months']
        df['Charges_Evolution'] = df['Monthly Charges'] - df['Average_Monthly_Charges']
        
        # Service usage features
        service_columns = ['Phone Service', 'Internet Service', 'Online Security',
                          'Online Backup', 'Device Protection', 'Tech Support',
                          'Streaming TV', 'Streaming Movies']
        
        df['Total_Services'] = df[service_columns].apply(
            lambda x: x.str.count('Yes').sum() if x.dtype == 'object' else x.sum(), axis=1
        )
        
        # Customer segments
        df['Value_Segment'] = pd.qcut(df['Monthly Charges'], q=4, 
                                    labels=['Low', 'Medium', 'High', 'Premium'])
        # ... your training code ...
        bins = pd.qcut(df['Monthly Charges'], q=4, retbins=True)[1] # Get the bin edges
        joblib.dump(bins, 'quantile_bins.pkl')  # Save the bins
        
        return df
    
    def engineer_advanced_features(self, df):
        """Create more sophisticated features."""
        df = df.copy()
        
        # Contract risk score
        contract_risk = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
        df['Contract_Risk_Score'] = df['Contract'].map(contract_risk)
        
        # Payment reliability
        payment_risk = {
            'Electronic check': 3,
            'Mailed check': 2,
            'Bank transfer (automatic)': 1,
            'Credit card (automatic)': 1
        }
        df['Payment_Risk_Score'] = df['Payment Method'].map(payment_risk)
        
        # Service dependency score
        service_weights = {
            'Phone Service': 1,
            'Internet Service': 2,
            'Online Security': 0.5,
            'Online Backup': 0.5,
            'Device Protection': 0.5,
            'Tech Support': 0.5,
            'Streaming TV': 1,
            'Streaming Movies': 1
        }
        
        df['Service_Dependency_Score'] = sum(
            (df[service] == 'Yes').astype(int) * weight
            for service, weight in service_weights.items()
        )
        
        # Loyalty-adjusted value
        df['Loyalty_Adjusted_Value'] = (
            df['Monthly Charges'] * np.log1p(df['Tenure Months'])
        )
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical variables with proper handling."""
        df = df.copy()
        
        # Features for label encoding
        label_encode_cols = ['Gender', 'Contract', 'Payment Method']
        
        # Features for one-hot encoding
        onehot_cols = ['Internet Service', 'Value_Segment']
        
        # Label encoding
        for col in label_encode_cols:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_Encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        # One-hot encoding
        df = pd.get_dummies(df, columns=onehot_cols, prefix=onehot_cols)
        
        return df
    
    def scale_numerical_features(self, df):
        """Scale numerical features with proper handling of skewness."""
        df = df.copy()
        
        # Basic numerical features
        basic_num_cols = ['Monthly Charges', 'Total Charges', 'Tenure Months']
        
        # Derived numerical features
        derived_num_cols = ['Revenue_per_Month', 'Average_Monthly_Charges',
                          'Charges_Evolution', 'Service_Dependency_Score',
                          'Loyalty_Adjusted_Value']
        
        all_num_cols = [col for col in basic_num_cols + derived_num_cols 
                       if col in df.columns]
        
        # Handle missing values
        df[all_num_cols] = self.imputer.fit_transform(df[all_num_cols])
        
        # Apply power transform for heavily skewed features
        df[all_num_cols] = self.power_transformer.fit_transform(df[all_num_cols])
        
        # Standard scaling
        df[all_num_cols] = self.scaler.fit_transform(df[all_num_cols])
        
        return df
    
    def prepare_data(self, file_path):
        """Complete data preparation pipeline."""
        # Load and validate
        df = self.load_data(file_path)
        validation_report = self.validate_data(df)
        
        if validation_report['duplicates'] > 0:
            logger.warning(f"Found {validation_report['duplicates']} duplicate rows")
            df = df.drop_duplicates()
        
        # Feature engineering
        df = self.engineer_basic_features(df)
        df = self.engineer_advanced_features(df)
        
        # Encoding and scaling
        df = self.encode_categorical_features(df)
        df = self.scale_numerical_features(df)
        
        logger.info("Data preparation completed successfully")
        return df, validation_report

if __name__ == "__main__":
    prep = DataPreparation()
    processed_df, validation_report = prep.prepare_data('./data/Telco_customer_churn_with_text.csv')
    processed_df.to_csv('./data/processed_telco_data.csv', index=False)