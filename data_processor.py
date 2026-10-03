import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    """Advanced Data Processor for Epidemiological Datasets."""
    def __init__(self, file_path):
        self.file_path = file_path
        self.le_disease = LabelEncoder()
        self.le_county  = LabelEncoder()
        self.le_sex     = LabelEncoder()

    def load_and_clean(self):
        # Load dataset
        df = pd.read_csv(self.file_path).fillna(0)
        df.columns = df.columns.str.strip()
        
        # Sort for time-series feature engineering
        df = df.sort_values(['country', 'Disease', 'Sex', 'Year'])
        
        # ── Feature Engineering ──
        # Calculate case growth compared to previous records
        df['Prev_Year_Count'] = df.groupby(['country', 'Disease', 'Sex'])['Count'].shift(1).fillna(0)
        df['Growth_Rate']     = df['Count'] - df['Prev_Year_Count']
        
        # ── Define Proper Risk Levels ──
        # Using Epidemiological thresholds: 0=Low, 1=Medium, 2=High
        df['Risk_Level'] = np.where(df['Rate'] > 2.0, 2,  # High
                           np.where(df['Rate'] > 0.5, 1, 0)) # Medium else Low
        
        # ── Categorical Encoding ──
        df['Disease_enc'] = self.le_disease.fit_transform(df['Disease'])
        df['County_enc']  = self.le_county.fit_transform(df['country'])
        df['Sex_enc']     = self.le_sex.fit_transform(df['Sex'])
        
        return df

    def get_encoders(self):
        return {
            "disease": self.le_disease,
            "county":  self.le_county,
            "sex":     self.le_sex
        }
