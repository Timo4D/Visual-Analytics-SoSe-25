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
feature_names  = json.load(open('notebooks/artifacts/feature_names.json', 'r'))
iso_forest     = joblib.load('notebooks/artifacts/iso_forest.joblib')
shap_explainer = joblib.load('notebooks/artifacts/shap_explainer.joblib')
shap_values    = np.load('notebooks/artifacts/shap_values.npy', allow_pickle=True)
with open('notebooks/artifacts/lime_explainer.pkl', 'rb') as f:
    lime_explainer = dill.load(f)

# --- LOAD DATA ---
df = pd.read_csv("notebooks/artifacts/processed_data.csv")
# # Select a random sample from the dataframe
# random_sample = df.sample(n=1)
# X_sample = random_sample[feature_names]


tabs = st.tabs([
    "1. Datenübersicht",
    "2. Modellevaluation", 
])

# --- TAB 1: Datenübersicht ---
with tabs[0]:
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Datenübersicht")
        # st.dataframe(df.head(10), use_container_width=True)
        st.dataframe(df, use_container_width=True)
        # Dropdown to select row index
        selected_idx = st.selectbox("Wähle eine Instanz aus", df.index)
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"Anzahl der Zeilen: {df.shape[0]}")
        with c2:
            st.write(f"Anzahl der Spalten: {df.shape[1]}")
        
    
    with col2: 
        st.subheader("Streudiagramm")

        c1, c2 = st.columns(2)
        with c1:
            x_var = st.selectbox("X-Achse", df.columns, index=df.columns.get_loc("LeadTime"))
        with c2:
            y_var = st.selectbox("Y-Achse", df.columns, index=df.columns.get_loc("StaysInWeekNights"))
        color_var = "IsCanceled" 
        
        # Create the main scatter plot
        fig = px.scatter(df, x=x_var, y=y_var, color=df[color_var].astype(str),
                title=f"{x_var} vs {y_var} – gefärbt nach {color_var}")
        
        # Add the selected instance as a highlighted point in red
        selected_row = df.loc[selected_idx]
        fig.add_scatter(
            x=[selected_row[x_var]], 
            y=[selected_row[y_var]], 
            mode='markers',
            marker=dict(color='red', size=12, symbol='diamond', line=dict(color='darkred', width=2)),
            name='Selected Instance',
            showlegend=True
        )
        
        # Move legend below the plot
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col3:
        st.subheader("Erklärungen")
        c1, c2 = st.columns(2)
        with c1: 
            st.metric("Actual", "Anomaly" if df.loc[selected_idx, "IsCanceled"] else "Normal")
        with c2:
            st.metric("Prediction", "Anomaly" if iso_forest.predict(df.drop(columns=['IsCanceled']).iloc[[selected_idx]])[0] == -1 else "Normal")
        
        df_features = df.drop(columns=['IsCanceled'])
        instance = df_features.loc[selected_idx]
        
        
        st.subheader("SHAP Erklärungen")
        shap_fig = shap.force_plot(
            shap_explainer.expected_value,
            shap_values[selected_idx],
            instance,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        st.pyplot(shap_fig)

        st.subheader("LIME Erklärungen")
        def predict_proba(X):
            raw = iso_forest.decision_function(X)
            norm = (raw - raw.min()) / (raw.max() - raw.min())
            return np.vstack([1 - norm, norm]).T

        lime_exp = lime_explainer.explain_instance(
            instance.values,
            predict_proba,
            num_features=10
        )
        st.pyplot(lime_exp.as_pyplot_figure())
        
    with tabs[1]:
        st.subheader("Datensatz Übersicht")
        
        # Use IsCanceled instead of Anomaly
        anomalies = df[df["IsCanceled"] == 1]  # Assuming 1 represents anomalies/cancellations
        normals = df[df["IsCanceled"] == 0]    # Assuming 0 represents normal/non-cancellations
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Anomalien (Stornierungen)", anomalies.shape[0])
        c2.metric("Normale (Keine Stornierung)", normals.shape[0])
        c3.metric("Anteil Anomalien", f"{anomalies.shape[0]/len(df):.2%}")
        
        st.markdown("---")
        st.subheader("Modell-Performance")
        
        try:
            with open("notebooks/artifacts/model_metrics.json", "r") as f:
                metrics = json.load(f)

            # Runde die Metriken auf 3 Nachkommastellen
            for k in ["balanced_accuracy", "precision", "recall", "f1_score"]:
                if k in metrics and isinstance(metrics[k], float):
                    metrics[k] = round(metrics[k], 3)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Balanced Accuracy", metrics.get("balanced_accuracy", "N/A"))
            c2.metric("Precision", metrics.get("precision", "N/A"))
            c3.metric("Recall", metrics.get("recall", "N/A"))
            c4.metric("F1-Score", metrics.get("f1_score", "N/A"))
            
            st.markdown("**Confusion Matrix**")
            try:
                st.image("notebooks/artifacts/confusion_matrix.png")
            except FileNotFoundError:
                st.warning("Confusion Matrix Bild nicht gefunden: notebooks/artifacts/confusion_matrix.png")
                
        except FileNotFoundError:
            st.warning("Modell-Metriken nicht gefunden: notebooks/artifacts/model_metrics.json")
            st.info("Bitte stellen Sie sicher, dass die Datei existiert und die Metriken korrekt gespeichert wurden.")