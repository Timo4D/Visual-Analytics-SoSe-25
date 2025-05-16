import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px
import shap
import lime.lime_tabular

# --- PAGE SETUP ---
st.set_page_config(layout="wide")
st.title("🧠 XAI-Dashboard zur Anomalieerkennung bei Hotelbuchungen")

# --- LOAD DATA ---
df = pd.read_csv("./data/H1.csv")
df = df.drop(columns=['Company', 'Agent', 'ReservationStatusDate'], errors='ignore')

# --- ENCODING ---
cat_cols = df.select_dtypes(include='object').columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# --- SCALING + PCA ---
target = df['IsCanceled']
features = df.drop(columns=['IsCanceled'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

pca_2d = PCA(n_components=2).fit_transform(X_scaled)
pca_3d = PCA(n_components=3).fit_transform(X_scaled)

df['PCA1'] = pca_2d[:, 0]
df['PCA2'] = pca_2d[:, 1]
df['PCA3'] = pca_3d[:, 2]

# --- ANOMALY DETECTION mit Train-Test-Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, target, test_size=0.3, random_state=42, stratify=target
)

iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X_train)

# Anomalie-Prediction auf Testdaten
y_pred = iso_forest.predict(X_test)
y_pred = np.where(y_pred == -1, 1, 0)

# Evaluation (Optional im Dashboard)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Predict für alle
df['Anomaly'] = iso_forest.predict(X_scaled)
df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1}).astype(int)

shap_explainer = shap.TreeExplainer(iso_forest)
shap_values = shap_explainer.shap_values(X_scaled)

left, right = st.columns(2)

# --- UI: TABS ---
#tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Scatterplot", "🧪 Modellevaluation", "📌 Feature Importance", "🔎 SHAP", "🟢 LIME"])

# === TAB 1: SCATTERPLOT ===
with left:
    st.header("Scatterplot mit Zielvariable oder Anomalien")
    
    col1, col2 = st.columns(2)
    with col1:
        variable = st.selectbox("Exogene Variable auswählen", df.columns)
    with col2:
        mode = st.radio("Färbung", ["Zielvariable (IsCanceled)", "Anomalien (Isolation Forest)"])

    color_var = 'IsCanceled' if mode == "Zielvariable (IsCanceled)" else 'Anomaly'
    fig = px.scatter(df, x='PCA1', y='PCA2', color=df[color_var].astype(str), title=f'Scatterplot gefärbt nach {color_var}')
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Scattermatrix mit Anomalien anzeigen"):
        fig = sns.pairplot(df, hue='Anomaly', vars=['LeadTime', 'ADR', 'StaysInWeekendNights', 'TotalOfSpecialRequests'])
        st.pyplot(fig)

# === TAB 2: MODELLEVALUATION ===
with right:
    st.header("Modellevaluation – Isolation Forest")

    anomalies = df[df['Anomaly'] == 1]
    normals = df[df['Anomaly'] == 0]

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Anomalien", f"{anomalies.shape[0]}")
    col2.metric("📉 Normale", f"{normals.shape[0]}")
    col3.metric("📊 Anteil Anomalien", f"{(anomalies.shape[0] / df.shape[0]):.2%}")

    st.subheader("Boxplot-Vergleich")
    fig, ax = plt.subplots()
    sns.boxplot(x="Anomaly", y="LeadTime", data=df, ax=ax)
    st.pyplot(fig)


# === TAB 4: SHAP ===

st.header("SHAP Analyse")
anomaly_index = df[df['Anomaly'] == 1].index[0]

st.subheader("SHAP Force Plot")
fig = shap.force_plot(shap_explainer.expected_value, shap_values[anomaly_index, :], features.columns, matplotlib=True, show=False)
plt.tight_layout()
st.pyplot(plt.gcf())



left2, right2 = st.columns(2)

    # === TAB 3: FEATURE IMPORTANCE ===
with left2:
    st.header("Feature Importance Übersicht")
    st.markdown("🔍 Vergleich von Features, die für Modellentscheidungen wichtig sind.")

    shap_explainer = shap.TreeExplainer(iso_forest)
    shap_values = shap_explainer.shap_values(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, features, feature_names=features.columns, show=False)
    st.pyplot(fig)


# === TAB 5: LIME ===
with right2:
    st.header("LIME Analyse")
    st.write("Lokale Erklärung eines Ausreißers")

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_scaled,
        feature_names=features.columns.tolist(),
        class_names=['Normal', 'Anomaly'],
        mode='classification',
        verbose=False,
        random_state=42
    )

    def predict_proba_isoforest(X):
        preds = iso_forest.decision_function(X)
        preds = (preds - preds.min()) / (preds.max() - preds.min())
        preds = np.vstack([1 - preds, preds]).T
        return preds

    lime_exp = lime_explainer.explain_instance(X_scaled[anomaly_index], predict_proba_isoforest, num_features=10)
    fig = lime_exp.as_pyplot_figure()
    st.pyplot(fig)

    st.subheader("Top Features laut LIME")
    for feature, importance in lime_exp.as_list():
        st.write(f"{feature}: {importance:.4f}")
