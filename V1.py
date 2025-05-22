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
st.set_page_config(layout="wide")
st.title("🧠 XAI-Dashboard zur Anomalieerkennung bei Hotelbuchungen")

# --- LOAD ARTIFACTS ---
encoders       = joblib.load('notebooks/artifacts/encoders.joblib')
feature_names  = json.load(open('notebooks/artifacts/feature_names.json', 'r'))
scaler         = joblib.load('notebooks/artifacts/scaler.joblib')
pca2           = joblib.load('notebooks/artifacts/pca2.joblib')
pca3           = joblib.load('notebooks/artifacts/pca3.joblib')
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

# --- SCALE & PCA ---
X_scaled = scaler.transform(df[feature_names])
df['PCA1'], df['PCA2'] = pca2.transform(X_scaled).T
df['PCA3'] = pca3.transform(X_scaled)[:, 2]

# --- ANOMALY PREDICTION ---
df['Anomaly'] = (iso_forest.predict(X_scaled) == -1).astype(int)

# --- UI TABS ---
left, right = st.columns(2)

with left:
    st.header("Scatterplot mit Zielvariable oder Anomalien")
    col1, col2 = st.columns(2)
    with col1:
        variable = st.selectbox("Exogene Variable auswählen", df.columns)
    with col2:
        mode = st.radio("Färbung", ["Zielvariable (IsCanceled)", "Anomalien (Isolation Forest)"])
    color_var = 'IsCanceled' if mode.startswith('Zielvariable') else 'Anomaly'
    fig = px.scatter(df, x='PCA1', y='PCA2', color=df[color_var].astype(str), title=f'Scatterplot gefärbt nach {color_var}')
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.header("Modellevaluation – Isolation Forest")
    anomalies = df[df['Anomaly'] == 1]
    normals = df[df['Anomaly'] == 0]
    c1, c2, c3 = st.columns(3)
    c1.metric("📈 Anomalien", f"{anomalies.shape[0]}")
    c2.metric("📉 Normale", f"{normals.shape[0]}")
    c3.metric("📊 Anteil Anomalien", f"{(anomalies.shape[0] / df.shape[0]):.2%}")
    st.subheader("Boxplot-Vergleich")
    fig, ax = plt.subplots()
    sns.boxplot(x="Anomaly", y="LeadTime", data=df, ax=ax)
    st.pyplot(fig)

# SHAP ANALYSIS
st.header("SHAP Analyse")
anomaly_index = df[df['Anomaly'] == 1].index[0]
fig = shap.force_plot(
    shap_explainer.expected_value,
    shap_values[anomaly_index, :],
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.tight_layout()
st.pyplot(plt.gcf())

# FEATURE IMPORTANCE
tab1, tab2 = st.tabs(["Feature Importance", "LIME"])
with tab1:
    st.header("Feature Importance Übersicht")
    fig, ax = plt.subplots(figsize=(10,6))
    shap.summary_plot(shap_values, pd.DataFrame(X_scaled, columns=feature_names), feature_names=feature_names, show=False)
    st.pyplot(fig)

    
# LIME ANALYSIS
with tab2:
    st.header("LIME Analyse")
    def predict_proba(X):
        preds = iso_forest.decision_function(X)
        preds = (preds - preds.min())/(preds.max()-preds.min())
        return np.vstack([1-preds, preds]).T

    lime_exp = lime_explainer.explain_instance(
        X_scaled[anomaly_index],
        predict_proba,
        num_features=10
    )
    fig = lime_exp.as_pyplot_figure()
    st.pyplot(fig)

    st.subheader("Top Features laut LIME")
    for feature, importance in lime_exp.as_list():
        st.write(f"{feature}: {importance:.4f}")
