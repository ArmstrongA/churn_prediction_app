# data_preparation3.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
import joblib
import logging
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline 
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparation:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer()
        self.imputer = SimpleImputer(strategy='mean')
        
    def load_data(self, data):
        """Load and validate the dataset."""
        try:
            df = data
            
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
        bins = joblib.load('quantile_bins.pkl') # Load the bins
        df['Value_Segment'] = pd.cut(df['Monthly Charges'], bins=bins, include_lowest=True, labels=['Low', 'Medium', 'High', 'Premium'])
        
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
    
    def extract_sentiment(self, text_column, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """Extracts sentiment from a text column and returns the updated DataFrame.
        
        Args:
            text_column (pd.Series): Column containing text data
            model_name (str): Hugging Face model name for sentiment analysis
            
        Returns:
            pd.Series: Series containing sentiment labels
        """
        # Load model and tokenizer once
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Function to get sentiment for a single text
        def get_sentiment(text):
            # Handle NaN values gracefully
            if pd.isna(text) or text.strip() == "":
                return "Neutral"
            
            # Tokenize and get model outputs
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get predicted label
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()
            return model.config.id2label[predicted_class_id]
        
        # Apply sentiment extraction to the text column
        return text_column.apply(get_sentiment)
    def extract_and_reduce_features(self, text_column, model_name="sentence-transformers/all-mpnet-base-v2", n_components=10):
        """
        Extracts text features using Hugging Face model, performs PCA to reduce dimensions,
        and returns a DataFrame with the reduced features.
        """
        # Load feature extraction pipeline once
        feature_extractor = pipeline(
            "feature-extraction",
            model=model_name,
            framework="pt"
        )
        
        # Function to extract features for one text
        def get_features(text):
            # Handle NaN values gracefully
            if pd.isna(text) or text.strip() == "":
                return np.zeros((768,))  # Return zero vector for empty text
            
            # Extract features and take mean across tokens
            features = feature_extractor(text, return_tensors="pt")[0]
            reduced_features = features.numpy().mean(axis=0)
            return reduced_features
        
        # Apply feature extraction to the text column
        feature_matrix = np.stack(text_column.apply(get_features))
        
        # # Perform PCA to reduce the dimensionality to n_components
        # pca = PCA(n_components=n_components)
        # reduced_features = pca.fit_transform(feature_matrix)
        
        # # Create DataFrame for the reduced features
        # feature_columns = [f'pca_feature_{i}' for i in range(n_components)]
        # feature_df = pd.DataFrame(reduced_features, columns=feature_columns)
        # Perform Random Projection to reduce the dimensionality to n_components
        random_projection = GaussianRandomProjection(n_components=n_components, random_state=42)
        reduced_features = random_projection.fit_transform(feature_matrix)

        # Create DataFrame for the reduced features
        feature_columns = [f'pca_{i}' for i in range(n_components)]
        feature_df = pd.DataFrame(reduced_features, columns=feature_columns)
        
        return feature_df

    def prepare_data(self, data):
        """Complete data preparation pipeline."""
        # Load and validate
        df = self.load_data(data)
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

        # Add text features
        df['customer_sentiment'] = self.extract_sentiment(df['customer_text'])
        label_encoder = LabelEncoder()
        df['sentiment_encoded'] = label_encoder.fit_transform(df['customer_sentiment'])
        # Extract features and reduce them to 10 components
        reduced_features_df = self.extract_and_reduce_features(df['customer_text'])

        # Concatenate the reduced features with the original DataFrame
        df = pd.concat([df, reduced_features_df], axis=1)
        
        logger.info("Data preparation completed successfully")
        return df, validation_report
    

if __name__ == "__main__":
    prep = DataPreparation()
    processed_df, validation_report = prep.prepare_data()
    # Use only in training
    # processed_df.to_csv('.data/model_data.csv', index=False) 