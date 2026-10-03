import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from data_processor import DataProcessor
from model_engine import ModelEngine

# ─── THEME & PAGE CONFIG ───
st.set_page_config(page_title="Public Health Intel", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { border: 1px solid #e0e0e0; padding: 20px; border-radius: 8px; background-color: white; }
    h1 { color: #1a365d; }
    .section-header { background-color: #1a365d; color: white; padding: 10px; border-radius: 5px; margin-top: 20px; }
    .prediction-card {
        background: linear-gradient(135deg, #ffffff 0%, #f1f4f8 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 8px solid #1a365d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .risk-high { border-left-color: #d9534f; }
    .risk-medium { border-left-color: #f0ad4e; }
    .risk-low { border-left-color: #5cb85c; }
    </style>
    """, unsafe_allow_html=True)

# ─── INITIALIZATION ───
@st.cache_resource
def setup_system():
    dp = DataProcessor("Infectious_Disease_cases2001-2014.csv")
    df = dp.load_and_clean()
    
    me = ModelEngine()
    accuracy = me.train_and_evaluate(df)
    
    return df, me, dp, accuracy

df, engine, processor, model_acc = setup_system()

# ─── HEADER ───
st.title("Disease Trend analysis and outbreak system")
st.markdown("---")

# ─── DASHBOARD CONTENT ───
tabs = st.tabs(["📊 Global Overview", "🔬 Deep EDA Analysis", "🎯 Prediction AI", "🌍 County/Sex Analysis", "🛰️ Outbreak Zone Detection", "🔮 Future Forecaster"])

# --- TAB 1: GLOBAL OVERVIEW ---
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cases", f"{df['Count'].sum():,.0f}")
    c2.metric("Total Population", f"{df['Population'].max():,}")
    c3.metric("Diseases Tracked", df['Disease'].nunique())
    c4.metric("Avg Infection Rate", f"{df['Rate'].mean():.2f}")
    
    st.markdown("---")
    st.subheader("📈 Case Count Trends (2001–2014)")
    dis_select = st.selectbox("Disease to Visualize:", sorted(df['Disease'].unique()))
    trend_df = df[df['Disease'] == dis_select].groupby('Year')['Count'].sum()
    st.area_chart(trend_df, color="#2c5282")

# --- TAB 2: DEEP EDA ANALYSIS (NEW GRAPHS) ---
with tabs[1]:
    st.markdown('<div class="section-header">🔍 Deep Data Exploratory Analysis</div>', unsafe_allow_html=True)
    
    row1_c1, row1_c2 = st.columns(2)
    
    with row1_c1:
        st.subheader("🔥 Feature Correlation Heatmap")
        fig, ax = plt.subplots()
        corr = df[['Year', 'Count', 'Population', 'Rate', 'Risk_Level']].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
        st.caption("Shows how different factors like Population and Rate relate to Risk.")

    with row1_c2:
        st.subheader("📊 Distribution of Infection Rates")
        fig, ax = plt.subplots()
        sns.histplot(df['Rate'], bins=50, kde=True, color="#d9534f", ax=ax)
        ax.set_xlim(0, 10) # Clipping outliers for better visibility
        st.pyplot(fig)
        st.caption("Most records show low rates, with a long tail of high-risk outliers.")

    st.markdown("---")
    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        st.subheader("🍕 Case Distribution by Sex")
        sex_counts = df.groupby('Sex')['Count'].sum().drop('Total', errors='ignore')
        fig, ax = plt.subplots()
        ax.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', startangle=90, colors=["#ff9999","#66b3ff"])
        st.pyplot(fig)

    with row2_c2:
        st.subheader("🦠 Top 10 High-Prevalence Diseases")
        top_10_dis = df.groupby('Disease')['Count'].sum().nlargest(10).sort_values()
        st.bar_chart(top_10_dis, color="#5cb85c")

    st.markdown("---")
    st.subheader("📍 Population vs Case Count (Density)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df[df['Count'] > 0], x="Population", y="Count", hue="Risk_Level", alpha=0.5, palette="viridis", ax=ax)
    st.pyplot(fig)

# --- TAB 3: PREDICTION AI ---
with tabs[2]:
    st.subheader("🤖 Outbreak Risk AI Assessment")
    with st.form("risk_ai_form"):
        col_a, col_b, col_c = st.columns(3)
        p_year    = col_a.selectbox("Analysis Year", sorted(df['Year'].unique(), reverse=True))
        p_disease = col_b.selectbox("Target Disease", sorted(df['Disease'].unique()))
        p_county  = col_c.selectbox("County/Region", sorted(df['country'].unique()))
        
        col_d, col_e = st.columns(2)
        p_sex     = col_d.selectbox("Demographic group", sorted(df['Sex'].unique()))
        p_rate    = col_e.number_input("Input Infection Rate", value=0.5, step=0.1)
        
        # Derived inputs
        p_pop   = df[df['country'] == p_county]['Population'].mean()
        p_count = (p_rate * p_pop) / 100000
        p_prev  = p_count * 0.95
        p_growth = p_count - p_prev

        if st.form_submit_button("🚀 Run AI Analysis"):
            encoders = processor.get_encoders()
            d_enc = encoders['disease'].transform([p_disease])[0]
            c_enc = encoders['county'].transform([p_county])[0]
            s_enc = encoders['sex'].transform([p_sex])[0]
            
            input_features = [p_year, p_pop, p_count, p_rate, d_enc, c_enc, s_enc, p_prev, p_growth]
            risk_label, confidence = engine.predict_risk(input_features)
            
            st.markdown("---")
            st.info(f"### Predicted Risk: {risk_label}")
            st.write(f"Confidence Level: **{confidence}%**")

# --- TAB 4: COUNTY/SEX ANALYSIS ---
with tabs[3]:
    st.subheader("🔥 Disease Spread Heatmap (Top 10 Counties)")
    top_counties = df.groupby('country')['Count'].sum().nlargest(10).index
    heatmap_data = df[df['country'].isin(top_counties)].pivot_table(
        index='country', columns='Year', values='Rate', aggfunc='mean'
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt=".1f", ax=ax)
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("📈 Yearly Case Count Trend by Sex")
    # Group by Year and Sex to see trends
    trend_sex_df = df[df['Sex'] != 'Total'].groupby(['Year', 'Sex'])['Count'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=trend_sex_df, x='Year', y='Count', hue='Sex', marker='o', palette="magma", ax=ax)
    ax.set_title("Evolution of Total Cases: Male vs Female")
    ax.set_ylabel("Total Cases")
    st.pyplot(fig)
    st.caption("This trend line highlights the divergence or convergence of reported cases between sexes over the study period.")

    st.subheader("⚠️ Risk Level Distribution by Sex")
    risk_sex_df = df[df['Sex'] != 'Total'].groupby(['Sex', 'Risk_Level']).size().unstack(fill_value=0)
    # Map risk levels to labels
    risk_sex_df.columns = ['Low (0)', 'Medium (1)', 'High (2)']
    fig, ax = plt.subplots()
    risk_sex_df.plot(kind='bar', stacked=True, color=['#5cb85c', '#f0ad4e', '#d9534f'], ax=ax)
    ax.set_ylabel("Count of Records")
    ax.set_title("Epidemiological Risk Severity by Demographic")
    st.pyplot(fig)
    st.caption("Compares how risk severity is distributed across Male and Female populations.")

# --- TAB 5: OUTBREAK ZONE DETECTION ---
with tabs[4]:
    st.markdown('<div class="section-header">🛰️ Automated Outbreak Detection & Hotspot Analysis</div>', unsafe_allow_html=True)
    
    # ── HOTSPOT DETECTION (Top 5 Counties by Rate) ──
    latest_year = df['Year'].max()
    st.subheader(f"🚩 Current Outbreak Hotspots (Year {latest_year})")
    
    hotspots = df[df['Year'] == latest_year].groupby('country')['Rate'].mean().nlargest(5).reset_index()
    
    h_col1, h_col2 = st.columns([1, 2])
    with h_col1:
        st.write("Top 5 Zones at Risk:")
        for idx, row in hotspots.iterrows():
            st.warning(f"**{row['country']}**: Rate {row['Rate']:.2f}")
            
    with h_col2:
        fig, ax = plt.subplots()
        sns.barplot(data=hotspots, x='Rate', y='country', palette='Reds_r', ax=ax)
        st.pyplot(fig)

    st.markdown("---")
    
    # ── GROWTH ALERTS (High Growth in Cases) ──
    st.subheader("🚀 Rapid Growth Alerts (Zonewise)")
    growth_df = df[df['Year'] == latest_year].groupby('country')['Growth_Rate'].sum().nlargest(5).reset_index()
    
    if not growth_df.empty and growth_df['Growth_Rate'].max() > 0:
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.error("Zones with highest case growth since previous year:")
            st.table(growth_df)
        with g_col2:
            st.info("These zones are showing rapid escalation of cases and require immediate medical intervention.")
    else:
        st.success("No rapid growth detected across zones for the latest recorded period.")

    st.markdown("---")
    
    # ── DISEASE DISTRIBUTION BY ZONE ──
    st.subheader("📊 Disease Distribution across Top Zones")
    top_zones = df.groupby('country')['Count'].sum().nlargest(8).index
    zone_dist = df[df['country'].isin(top_zones)].groupby(['country', 'Disease'])['Count'].sum().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    zone_dist.plot(kind='bar', stacked=True, ax=ax, cmap='tab20')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    st.pyplot(fig)
    st.caption("Detailed breakdown of which diseases are contributing to the case counts in the highest-density zones.")

# --- TAB 6: FUTURE FORECASTER ---
with tabs[5]:
    st.markdown('<div class="section-header">🔮 AI Future Outbreak Forecaster</div>', unsafe_allow_html=True)
    st.write("Project current trends into the future to anticipate potential outbreaks.")
    
    with st.expander("🔬 Model Intelligence & Architecture", expanded=False):
        st.write("""
        **This system uses a Hybrid AI Pipeline for future projections:**
        1. **Time-Series Engine**: Uses *Holt's Linear Trend (Exponential Smoothing)* to forecast future infection rates and counts by analyzing historical growth patterns.
        2. **Risk Classifier**: A *Random Forest Classifier* (trained with 95%+ accuracy) analyzes the forecasted data to categorize the Epidemiological Risk Level.
        """)
    
    with st.expander("⚙️ Forecast Settings", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        f_disease = f_col1.selectbox("Disease to Forecast", sorted(df['Disease'].unique()), key="f_dis")
        f_county  = f_col2.selectbox("County/Region", sorted(df['country'].unique()), key="f_cnt")
        f_sex     = f_col3.selectbox("Sex/Demographic", sorted(df['Sex'].unique()), key="f_sex")
        f_year    = f_col4.number_input("Target Year (e.g. 2022, 2035)", min_value=2015, max_value=2050, value=2024)

    if st.button("✨ Generate Future Projection", type="primary"):
        forecast_data, error = engine.forecast_future(df, f_disease, f_county, f_sex, f_year)
        
        if error:
            st.error(error)
        else:
            # Get Risk Level from the main classifier using forecasted data
            encoders = processor.get_encoders()
            d_enc = encoders['disease'].transform([f_disease])[0]
            c_enc = encoders['county'].transform([f_county])[0]
            s_enc = encoders['sex'].transform([f_sex])[0]
            
            input_features = [
                forecast_data['year'], 
                forecast_data['population'], 
                forecast_data['count'], 
                forecast_data['rate'], 
                d_enc, c_enc, s_enc, 
                forecast_data['prev_count'], 
                forecast_data['growth']
            ]
            
            risk_label, confidence = engine.predict_risk(input_features)
            
            # Risk color class
            risk_class = "risk-low"
            if "HIGH" in risk_label: risk_class = "risk-high"
            elif "MEDIUM" in risk_label: risk_class = "risk-medium"

            # --- Visualizing the Forecast with Plotly ---
            st.markdown("---")
            st.markdown(f"""
                <div class="prediction-card {risk_class}">
                    <h2 style='margin-top:0;'>Forecast Result for {f_year}</h2>
                    <p>Analysis for <b>{f_disease}</b> in <b>{f_county}</b> ({f_sex})</p>
                    <hr>
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <p style="font-size: 0.9rem; color: #666;">Predicted Cases</p>
                            <h3 style="margin:0;">{forecast_data['count']:,.0f}</h3>
                        </div>
                        <div>
                            <p style="font-size: 0.9rem; color: #666;">Infection Rate</p>
                            <h3 style="margin:0;">{forecast_data['rate']:.2f}</h3>
                        </div>
                        <div>
                            <p style="font-size: 0.9rem; color: #666;">Risk Assessment</p>
                            <h3 style="margin:0;">{risk_label}</h3>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Interactive Trend with Plotly
            import plotly.graph_objects as go
            hist_data = df[(df['Disease'] == f_disease) & (df['country'] == f_county) & (df['Sex'] == f_sex)].sort_values('Year')
            
            fig = go.Figure()
            # Historical
            fig.add_trace(go.Scatter(x=hist_data['Year'], y=hist_data['Rate'], mode='lines+markers', name='Historical', line=dict(color='#1a365d', width=3)))
            # Forecast
            fig.add_trace(go.Scatter(x=[hist_data['Year'].iloc[-1], f_year], y=[hist_data['Rate'].iloc[-1], forecast_data['rate']], 
                                     mode='lines+markers', name='AI Projection', line=dict(color='#d9534f', dash='dash', width=3),
                                     marker=dict(size=[0, 12], color='#d9534f')))
            
            fig.update_layout(title=f"Evolution & Projection: {f_disease} Infection Rate", xaxis_title="Year", yaxis_title="Rate (per 100k)",
                              template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            # --- Dynamic Horizon Insight ---
            st.info(f"📍 **Dynamic Forecasting Horizon:** The AI is currently extrapolating trends from 2001-2014 to the year {f_year}. While the time-series model is statistically robust, projections for years further than 10 years out (e.g. beyond 2024) should be treated as hypothetical scenarios for long-term planning.")
            

st.markdown("---")
