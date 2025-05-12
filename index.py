import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import streamlit as st


st.title("Ein interaktives explainable AI (XAI) Dashboard zur Anomalieerkennung bei Hotelbuchungen")
st.markdown("""
            Kilian Mütz (79561) und Timo Gerstenhauer (86164) 

            ## Business Understanding
            - "Hotel booking demand dataset"
            - Fokus auf das Untersuchungsobjekt "Resort-Hotel H1" an der Algravenküste Portugals
            - Der Datensatz umfasst den Zeitraum vom 01.07.2015 bis zum 31.08.2017
            
            ## Data Understanding
            - Der Datensatz wurde als CSV "H1" geladen
            - Es existieren 31 Variablen mit ~40.000 Beobachtungen
""")


st.header("Daten")
df = pd.read_csv("data/H1.csv")
st.write("Shape des Datensatzes: ", df.shape)
st.dataframe(df)
st.write("Datatypes des Datensatzes: ", df.dtypes)


st.header("Datenvorverarbeitung")
st.write("""Wir entfernen die Spalten 'Company', 'Agent' und 'ReservationStatusDate' aus dem Datensatz, da sie für unsere Analyse nicht relevant sind.
Wir kodieren die kategorischen Variablen mit Label Encoding, um sie für die Modellierung vorzubereiten.
Wir trennen die Zielvariable 'IsCanceled' von den Features.
Wir verwenden Label Encoding für die kategorischen Variablen und StandardScaler für die numerischen Variablen.
Wir verwenden PCA zur Dimensionsreduktion auf 2 Dimensionen.""")

df = df.drop(columns=['Company', 'Agent', 'ReservationStatusDate'], errors='ignore')
cat_cols = df.select_dtypes(include='object').columns.tolist()
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

target = df['IsCanceled']
features = df.drop(columns=['IsCanceled'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

st.subheader("PCA mit zwei Dimensionen")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'IsCanceled': target
})

st.scatter_chart(
    data=pca_df,
    x='PCA1',
    y='PCA2',
    color='IsCanceled',
    use_container_width=True,
    height=500
)

st.subheader("PCA mit drei Dimensionen")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'PCA3': X_pca[:, 2],
    'IsCanceled': target
})
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_df['PCA1'], pca_df['PCA2'], pca_df['PCA3'], c=pca_df['IsCanceled'], cmap='viridis')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
st.pyplot(fig)

sns.pairplot(pca_df, hue='IsCanceled', palette='coolwarm', 
             diag_kind='kde', plot_kws={'alpha': 0.6, 's': 5})

plt.suptitle('Pairplot of PCA Components', y=1.02, fontsize=16)
plt.tight_layout()
st.pyplot(plt.gcf())

st.header("Anomalieerkennung mit Isolation Forest")
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X_scaled)

df['Anomaly'] = iso_forest.predict(X_scaled)

df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})

# Create DataFrame for the scatter chart
anomaly_pca_df = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'Anomaly': df['Anomaly']
})

# Display scatter chart with Streamlit
st.subheader("PCA – Anomalien durch Isolation Forest")
st.scatter_chart(
    data=anomaly_pca_df,
    x='PCA1',
    y='PCA2',
    color='Anomaly',
    use_container_width=True,
    height=500
)

from mpl_toolkits.mplot3d import Axes3D

# Create 3D PCA visualization for anomalies
st.subheader("3D PCA – Anomalien durch Isolation Forest")
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                     c=df['Anomaly'], cmap='coolwarm', alpha=0.6)

ax.set_title("3D PCA – Anomalien durch Isolation Forest")
ax.set_xlabel("PCA Komponente 1")
ax.set_ylabel("PCA Komponente 2")
ax.set_zlabel("PCA Komponente 3")
legend1 = ax.legend(*scatter.legend_elements(), title="Anomaly (1=Outlier)")
ax.add_artist(legend1)
st.pyplot(fig)

st.header("Data Evaluation")
anomalies = df[df['Anomaly'] == 1]
normals = df[df['Anomaly'] == 0]

st.write(f"""Anzahl Anomalien: {anomalies.shape[0]} \n
         
Anzahl normale Buchungen: {normals.shape[0]}""")
st.write(anomalies.describe())

st.subheader("Vergleich der Anomalien mit den normalen Buchungen")
fig, ax = plt.subplots(figsize=(10,6))
sns.kdeplot(anomalies['LeadTime'], label='Anomalien', fill=True, ax=ax)
sns.kdeplot(normals['LeadTime'], label='Normale Buchungen', fill=True, ax=ax)
ax.set_title('Lead Time – Vergleich Anomalien vs. Normal')
ax.legend()
st.pyplot(fig)

st.subheader("Paarweise Beziehungen zwischen ausgewählten Merkmalen")
fig = sns.pairplot(df, hue='Anomaly', vars=['LeadTime', 'ADR', 'StaysInWeekendNights', 'TotalOfSpecialRequests'])
plt.suptitle('Verteilungen der wichtigsten Features nach Anomalie-Status', y=1.02)
st.pyplot(fig)


st.header("Interpretation")
st.write(f"""
         Anzahl Anomalien: {anomalies.shape[0]}
         Anzahl normale Buchungen: {normals.shape[0]}
         Anteil Anomalien: {anomalies.shape[0] / df.shape[0]:.2%}
""")

st.header("SHAP")
import shap
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X_scaled)

explainer = shap.TreeExplainer(iso_forest)
shap_values = explainer.shap_values(X_scaled)
anomaly_index = df[df['Anomaly'] == 1].index[0]
# Create a SHAP force plot and display it in Streamlit
st.subheader("SHAP Force Plot für eine Anomalie")
fig = plt.figure(figsize=(12, 5))
shap_plot = shap.force_plot(explainer.expected_value, 
                          shap_values[anomaly_index,:], 
                          features.columns, 
                          matplotlib=True, 
                          show=False)
plt.tight_layout()
st.pyplot(fig)

# Create a SHAP summary plot
st.subheader("SHAP Summary Plot")
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, features, feature_names=features.columns, show=False)
plt.tight_layout()
st.pyplot(fig)

st.header("LIME")
import lime
import lime.lime_tabular
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_scaled,
    feature_names=features.columns.tolist(),
    class_names=['Normal', 'Anomaly'],
    mode='classification',
    verbose=True,
    random_state=42
)
anomaly_index = df[df['Anomaly'] == 1].index[0]

# predict_proba muss simuliert werden, weil Isolation Forest nicht klassisch ist
def predict_proba_isoforest(X):
    preds = iso_forest.decision_function(X)
    preds = (preds - preds.min()) / (preds.max() - preds.min())  # Normierung auf [0,1]
    preds = np.vstack([1 - preds, preds]).T
    return preds


lime_exp = lime_explainer.explain_instance(
    X_scaled[anomaly_index],
    predict_proba_isoforest,
    num_features=10
)


# Create a matplotlib figure of the explanation
fig = lime_exp.as_pyplot_figure()
plt.title("LIME Explanation for Anomaly")
plt.tight_layout()
st.pyplot(fig)

# Display feature importance as a list in Streamlit
st.subheader("Feature importance (showing top 10):")
for feature, importance in lime_exp.as_list():
    st.write(f"{feature}: {importance:.4f}")