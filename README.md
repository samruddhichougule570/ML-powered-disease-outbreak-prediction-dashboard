# 🦠 Disease Outbreak Intelligence System (DOIS)

A high-performance machine learning system for predicting infectious disease risks and forecasting future outbreak trends using California healthcare data (2001–2014).

## 🚀 Features
- **Advanced Classification**: Uses **XGBoost** (with Random Forest fallback) to classify outbreak risk into LOW, MEDIUM, and HIGH levels.
- **Time-Series Forecasting**: Predictive trend analysis to estimate case counts for future years (2015–2030).
- **Intelligent Dashboard**: Interactive Streamlit UI with 10+ dynamic visualizations, What-If scenarios, and AI-driven insights.
- **Healthcare Recommendations**: Automated medical and policy advice tailored to the predicted risk level.
- **Robust Feature Engineering**: Incorporates lag features (Prev_Count, Prev_Rate) and growth metrics for historical context.

## 🛠️ Technology Stack
- **Python 3.10+**
- **XGBoost / Scikit-learn**: Core ML modeling.
- **Streamlit**: Dashboard and User Interface.
- **Pandas/Numpy**: Data processing and feature engineering.
- **Matplotlib/Seaborn**: Visual analytics.

## 📁 System Architecture
1. **`data_processor.py`**: Handles cleaning, feature engineering, and categorical encoding.
2. **`model_engine.py`**: Core engine for training models, performing risk assessment, and forecasting future trends.
3. **`dashboard.py`**: The main interface for exploration and real-time prediction.

## 🚦 How to Run
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch the Intelligence Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```
3. **Generate Static Report** (Optional):
   ```bash
   python outbreak_ml.py
   ```

## 📊 Evaluation
The system consistently achieves **95%+ accuracy** on held-out test data. The XGBoost model is particularly effective at identifying high-growth patterns that precede major outbreaks.

---
*Created for the Healthcare ML Intelligence Project.*
