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
st.set_page_config(layout="wide", menu_items={}, page_title="XAI Dashboard", page_icon="🏨")
st.title("🏨 Hotel Booking Anomalies")
#st.set_page_config(layout="wide")

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
])

# --- TAB 1: Datenübersicht ---
with tabs[0]:
    
    st.subheader("Streudiagramm")

    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("X-Achse", df.columns, index=df.columns.get_loc("LeadTime"))
    with col2:
        y_var = st.selectbox("Y-Achse", df.columns, index=df.columns.get_loc("StaysInWeekNights"))
    color_var = "IsCanceled" 
    fig = px.scatter(df, x=x_var, y=y_var, color=df[color_var].astype(str),
                     title=f"{x_var} vs {y_var} – gefärbt nach {color_var}")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Instanzanalyse")

    selected_indices = st.multiselect(
        "Wähle bis zu 5 Instanzen",
        options=df.index.tolist(),
        default=[df[df["Anomaly"] == 1].index[0]],
        max_selections=5
    )

    if selected_indices:
        for idx in selected_indices:
            inst = df.loc[idx]
            st.markdown(f"### Instanz {idx}")

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.markdown("**Ground Truth**")
                truth = "Anomaly" if inst["Anomaly"] else "Normal"
                st.write(truth)

            with col2:
                st.markdown("**Prediction**")
                pred = iso_forest.predict([X_scaled[idx]])[0]
                pred_label = "Anomaly" if pred == -1 else "Normal"
                correct = pred == -1 if inst["Anomaly"] else pred != -1
                st.write(f"{pred_label} – {'✔️ korrekt' if correct else '❌ falsch'}")

            with col3:
                st.markdown("**Feature-Werte**")
                st.dataframe(inst[feature_names].to_frame().T, use_container_width=True, height=120)

            shap_col, lime_col = st.columns(2)
            with shap_col:
                with st.expander("SHAP Summary (Feature-Importance)"):
                    st.markdown("**SHAP Force Plot**")
                    shap_fig = shap.force_plot(
                        shap_explainer.expected_value,
                        shap_values[idx],
                        feature_names=feature_names,
                        matplotlib=True,
                        show=False
                    )
                    st.pyplot(shap_fig)

            with lime_col:
                with st.expander("SHAP Summary (Feature-Importance)"):
                    st.markdown("**LIME Analyse**")
                    def predict_proba(X):
                        raw = iso_forest.decision_function(X)
                        norm = (raw - raw.min()) / (raw.max() - raw.min())
                        return np.vstack([1 - norm, norm]).T

                    lime_exp = lime_explainer.explain_instance(
                        X_scaled[idx],
                        predict_proba,
                        num_features=10
                    )
                    st.pyplot(lime_exp.as_pyplot_figure())

    # Unterer Bereich: Scatterplot
    

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

