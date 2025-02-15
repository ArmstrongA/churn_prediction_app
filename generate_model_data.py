import pandas as pd
from data_preparation import DataPreparation

# Load Data
prep = DataPreparation()
df = pd.read_csv("./data/Telco_customer_churn_with_text.csv")
new_df, validation_report = prep.prepare_data(df)
new_df.to_csv("./data/model_data.csv")