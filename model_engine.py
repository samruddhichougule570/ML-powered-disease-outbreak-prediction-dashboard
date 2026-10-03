import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import Holt
import warnings
warnings.filterwarnings('ignore')

class ModelEngine:
    """Robust AI engine for outbreak classification."""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        # Proper feature set including historical growth
        self.features = [
            'Year', 'Population', 'Count', 'Rate', 
            'Disease_enc', 'County_enc', 'Sex_enc',
            'Prev_Year_Count', 'Growth_Rate'
        ]
        self.risk_labels = {0: "🟢 LOW", 1: "🟡 MEDIUM", 2: "🔴 HIGH"}

    def train_and_evaluate(self, df):
        """Trains the model and returns performance metrics."""
        X = df[self.features]
        y = df['Risk_Level']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return round(acc * 100, 2)

    def predict_risk(self, data_list):
        """Predicts risk for a specific scenario."""
        # data_list should match the self.features order
        prediction = self.model.predict([data_list])[0]
        probs = self.model.predict_proba([data_list])[0]
        return self.risk_labels[prediction], round(max(probs) * 100, 2)

    def forecast_future(self, df, disease, county, sex, future_year):
        """Forecasts case count and rate for a future year using Exponential Smoothing (Holt's Linear Trend)."""
        # Filter data for specific disease/county/sex
        subset = df[(df['Disease'] == disease) & (df['country'] == county) & (df['Sex'] == sex)].sort_values('Year')
        
        if len(subset) < 3:
            return None, "Not enough (minimum 3) historical records for this specific cluster to perform time-series analysis."

        # Target variables
        target_cols = ['Rate', 'Count', 'Population']
        results = {}
        
        # Calculate steps from last year in data to target year
        last_year = subset['Year'].max()
        steps = future_year - last_year
        
        if steps <= 0:
            return None, f"Target year must be after the last recorded year ({last_year})."

        try:
            for col in target_cols:
                # Use Holt's Linear Trend (Exponential Smoothing with trend)
                model = Holt(subset[col].values, initialization_method="estimated").fit()
                forecast = model.forecast(steps)
                results[col.lower()] = max(0, forecast[-1])
            
            # Estimate prev_year for the growth feature
            prev_forecast = {}
            if steps > 1:
                for col in target_cols:
                    model = Holt(subset[col].values, initialization_method="estimated").fit()
                    f = model.forecast(steps - 1)
                    prev_forecast[col.lower()] = max(0, f[-1])
            else:
                # If steps=1, prev_year is just the last year from data
                for col in target_cols:
                    prev_forecast[col.lower()] = subset[col].iloc[-1]

            pred_growth = results['count'] - prev_forecast['count']

            return {
                "year": future_year,
                "rate": round(results['rate'], 2),
                "count": round(results['count'], 0),
                "population": round(results['population'], 0),
                "prev_count": round(prev_forecast['count'], 0),
                "growth": round(pred_growth, 2)
            }, None
            
        except Exception as e:
            return None, f"Time-series model error: {str(e)}"
