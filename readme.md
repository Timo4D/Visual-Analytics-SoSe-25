# Ein interaktives explainable AI (XAI) Dashboard zur Anomalieerkennung bei Hotelbuchungen
Kilian Mütz (79561) und Timo Gerstenhauer (86164) 

## Business Understanding
- "Hotel booking demand dataset"
- Fokus auf das Untersuchungsobjekt "Resort-Hotel H1" an der Algravenküste Portugals
- Der Datensatz umfasst den Zeitraum vom 01.07.2015 bis zum 31.08.2017

## Data Understanding
- Der Datensatz wurde als CSV "H1" geladen
- Es existieren 31 Variablen mit ~40.000 Beobachtungen

## Um das Dasboard zu starten

```bash
git clone https://github.com/Timo4D/Visual-Analytics-SoSe-25.git
cd Visual-Analytics-SoSe-25
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run Dashboard.py
```

-> http://localhost:8501/

-> Bei Problemen notebooks/precompute.ipynb durchlaufen lassen