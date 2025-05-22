# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json
import dill
import shap

# --- PAGE SETUP ---
st.set_page_config(layout="wide", menu_items={}, page_title="XAI Dashboard", page_icon="🧠")
st.title("🧠 XAI-Dashboard zur Anomalieerkennung bei Hotelbuchungen")

hide_streamlit_style = """
    <style>
    /* hide top header bar, menu icon, footer */
    header {visibility: hidden; height: 0; margin: 0; padding: 0;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden; height: 0; margin: 0; padding: 0;}

    /* collapse the automatic top padding Streamlit adds */
    div[data-testid="stAppViewContainer"] > div:first-child {padding-top: 0rem;}
    div[data-testid="stBlockContainer"] {padding-top: 0rem;}
    /* fallback for older versions */
    .block-container {padding-top: 0rem !important;}
    .css-18e3th9 {padding-top: 0rem !important;}
    </style>  
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- LOAD ARTIFACTS ---
encoders       = joblib.load('notebooks/artifacts/encoders.joblib')
feature_names  = json.load(open('notebooks/artifacts/feature_names.json', 'r'))
scaler         = joblib.load('notebooks/artifacts/scaler.joblib')
iso_forest     = joblib.load('notebooks/artifacts/iso_forest.joblib')
shap_explainer = joblib.load('notebooks/artifacts/shap_explainer.joblib')
shap_values    = np.load('notebooks/artifacts/shap_values.npy', allow_pickle=True)
with open('notebooks/artifacts/lime_explainer.pkl', 'rb') as f:
    lime_explainer = dill.load(f)

# --- LOAD DATA ---
df = pd.read_csv("./data/H1.csv")
df = df.drop(columns=['Company', 'Agent', 'ReservationStatusDate'], errors='ignore')

# --- APPLY ENCODING ---
for col, le in encoders.items():
    df[col] = le.transform(df[col].astype(str))

# --- SCALE ---
X_scaled = scaler.transform(df[feature_names])

# --- ANOMALY PREDICTION ---
df['Anomaly'] = (iso_forest.predict(X_scaled) == -1).astype(int)



tabs = st.tabs([
    "1. Datenübersicht",
    "2. Modellevaluation",
    "3. Erklärungen"  
])

# --- TAB 1: Datenübersicht ---
with tabs[0]:
    left, right = st.columns(2)
    with left:
        st.subheader("Streudiagramm")
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("X-Achse", df.columns, index=df.columns.get_loc("LeadTime"))
        with col2:
            y_var = st.selectbox("Y-Achse", df.columns, index=df.columns.get_loc("StaysInWeekNights"))
        mode = st.radio("Färbung", ["IsCanceled", "Anomalien"], horizontal=True)
        color_var = "IsCanceled" if mode=="IsCanceled" else "Anomaly"
        fig = px.scatter(df, x=x_var, y=y_var, color=df[color_var].astype(str),
                        title=f"{x_var} vs {y_var} – gefärbt nach {color_var}")
        
        st.plotly_chart(fig, use_container_width=True)
        
    with right:
        df

# --- TAB 2: Modellevaluation ---
with tabs[1]:
    st.subheader("Metric-Übersicht")
    anomalies = df[df["Anomaly"]==1]
    normals   = df[df["Anomaly"]==0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Anomalien", anomalies.shape[0])
    c2.metric("Normale",   normals.shape[0])
    c3.metric("Anteil Anomalien", f"{anomalies.shape[0]/len(df):.2%}")
    st.subheader("Boxplot LeadTime")
    fig, ax = plt.subplots()
    sns.boxplot(x="Anomaly", y="LeadTime", data=df, ax=ax)
    st.pyplot(fig)

with tabs[2]:
    st.header("Erklärungen")

    # SHAP Force-Plot
    with st.expander("SHAP Force-Plot"):
        anomaly_idx = df[df["Anomaly"]==1].index[0]
        fig = shap.force_plot(
            shap_explainer.expected_value,
            shap_values[anomaly_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig)

    # SHAP Summary (Feature Importance)
    with st.expander("SHAP Summary (Feature-Importance)"):
        fig, ax = plt.subplots(figsize=(10,6))
        shap.summary_plot(
            shap_values,
            pd.DataFrame(X_scaled, columns=feature_names),
            feature_names=feature_names,
            show=False
        )
        st.pyplot(fig)

    # LIME Analyse
    with st.expander("LIME Analyse"):
        def predict_proba(X):
            raw = iso_forest.decision_function(X)
            norm = (raw - raw.min())/(raw.max()-raw.min())
            return np.vstack([1-norm, norm]).T

        lime_exp = lime_explainer.explain_instance(
            X_scaled[anomaly_idx],
            predict_proba,
            num_features=10
        )
        st.pyplot(lime_exp.as_pyplot_figure())