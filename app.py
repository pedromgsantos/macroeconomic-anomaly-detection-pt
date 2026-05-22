# ===================================================================
# INTERACTIVE DASHBOARD - MACROECONOMIC ANOMALY ANALYSIS (Dynamic)
# ===================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL
from prophet import Prophet
import logging

# Reduce Prophet / cmdstanpy verbosity
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Macroeconomic Anomaly Monitoring",
    page_icon="🚨",
    layout="wide"
)

# --- CACHED FUNCTION: LOAD DATA AND RUN MODELS ---
@st.cache_data
def load_and_model_data():
    """
    Load processed quarterly data, rename columns, and run all anomaly detection models:
    - Isolation Forest (multivariate)
    - STL decomposition (univariate turning points)
    - Prophet (forecast-based anomalies on GDP)
    """
    # 1. Load processed data
    data_path = "data/dados_processados_trimestrais.csv"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.index.name = "date"

    # Original columns in the CSV:
    # Date, GDP_YoY_Growth, Total_Corporate_Credit, Total_Household_Credit, Total_Debt
    # Rename to shorter, more convenient names
    df.rename(
        columns={
            "GDP_YoY_Growth": "gdp",
            "Total_Corporate_Credit": "corporate_credit",
            "Total_Household_Credit": "household_credit",
            "Total_Debt": "total_debt",
        },
        inplace=True,
    )

    # --- 2. Run anomaly detection models ---

    # a) Isolation Forest (systemic / multivariate)
    features = ["gdp", "corporate_credit", "household_credit", "total_debt"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    iso_model = IsolationForest(
        n_estimators=200,
        contamination=0.05,  # relatively conservative
        random_state=42,
    )
    df["anomaly_isoforest_raw"] = iso_model.fit_predict(X_scaled)
    # Map: -1 = anomaly → 1, 1 = normal → 0
    df["anomaly_isoforest"] = df["anomaly_isoforest_raw"].apply(
        lambda x: 1 if x == -1 else 0
    )

    # b) STL decomposition (turning points / univariate)
    df["anomaly_stl"] = 0
    for col in features:
        stl = STL(df[col], period=4)  # quarterly data → period=4
        stl_result = stl.fit()
        residuals = stl_result.resid
        threshold = 2.5 * residuals.std()

        anomaly_idx = residuals[residuals.abs() > threshold].index
        df.loc[anomaly_idx, "anomaly_stl"] = 1

    # c) Prophet (forecast deviation, GDP only)
    df_prophet = (
        df[["gdp"]]
        .reset_index()
        .rename(columns={"date": "ds", "gdp": "y"})
    )

    prophet_model = Prophet(
        interval_width=0.95,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    prophet_model.fit(df_prophet)
    forecast = prophet_model.predict(df_prophet[["ds"]])
    df_forecast = forecast.set_index("ds")

    df["anomaly_prophet"] = (
        (df["gdp"] < df_forecast["yhat_lower"])
        | (df["gdp"] > df_forecast["yhat_upper"])
    ).astype(int)

    # 3. Count how many models flag an anomaly on each date
    df["anomaly_count"] = df[
        ["anomaly_isoforest", "anomaly_stl", "anomaly_prophet"]
    ].sum(axis=1)

    # Return final DataFrame without the raw Isolation Forest label
    return df.drop(columns=["anomaly_isoforest_raw"])


# --- LOAD AND MODEL DATA ---
# This is recomputed only if the CSV changes
df_final = load_and_model_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Visualisation Controls")

variable_labels = {
    "gdp": "GDP YoY Growth (%)",
    "corporate_credit": "Total Corporate Credit",
    "household_credit": "Total Household Credit",
    "total_debt": "Total Non-Financial Sector Debt",
}

selected_variable = st.sidebar.selectbox(
    "Select the time series to display:",
    options=list(variable_labels.keys()),
    format_func=lambda x: variable_labels[x],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Show anomalies from:**")
show_iso = st.sidebar.checkbox("Isolation Forest (systemic)", value=True)
show_stl = st.sidebar.checkbox("STL decomposition (turning points)", value=True)
show_prophet = st.sidebar.checkbox("Prophet (forecast deviation, GDP)", value=True)

# --- MAIN LAYOUT ---
st.title("🚨 Macroeconomic Anomaly Monitoring – Portugal")

# --- ALERT SYSTEM (LATEST QUARTER) ---
last_obs = df_final.iloc[-1]
if last_obs["anomaly_count"] > 0:
    st.error(
        f"ALERT: An anomaly is detected in the latest available quarter "
        f"({last_obs.name.strftime('%Y-%m-%d')})."
    )

st.markdown(
    "Interactive dashboard to explore anomalies in GDP, credit, and debt series for Portugal."
)

# --- MAIN CHART ---
st.header(f"Visual anomaly analysis: {variable_labels[selected_variable]}")

fig = go.Figure()

# Base time series line
fig.add_trace(
    go.Scatter(
        x=df_final.index,
        y=df_final[selected_variable],
        mode="lines",
        name=variable_labels[selected_variable],
        line=dict(color="lightgrey", width=2),
    )
)

# Isolation Forest anomalies
if show_iso:
    df_iso = df_final[df_final["anomaly_isoforest"] == 1]
    fig.add_trace(
        go.Scatter(
            x=df_iso.index,
            y=df_iso[selected_variable],
            mode="markers",
            name="Anomaly: Isolation Forest",
            marker=dict(color="red", size=10, symbol="circle"),
        )
    )

# STL anomalies
if show_stl:
    df_stl = df_final[df_final["anomaly_stl"] == 1]
    fig.add_trace(
        go.Scatter(
            x=df_stl.index,
            y=df_stl[selected_variable],
            mode="markers",
            name="Anomaly: STL",
            marker=dict(color="green", size=10, symbol="diamond"),
        )
    )

# Prophet anomalies (dates are GDP-based but shown on any series for consistency)
if show_prophet:
    df_prophet_anom = df_final[df_final["anomaly_prophet"] == 1]
    fig.add_trace(
        go.Scatter(
            x=df_prophet_anom.index,
            y=df_prophet_anom[selected_variable],
            mode="markers",
            name="Anomaly: Prophet",
            marker=dict(color="purple", size=10, symbol="x"),
        )
    )

# Consensus anomalies (2 or more models)
df_consensus = df_final[df_final["anomaly_count"] > 1]
fig.add_trace(
    go.Scatter(
        x=df_consensus.index,
        y=df_consensus[selected_variable],
        mode="markers",
        name="Consensus (≥ 2 models)",
        marker=dict(
            color="gold",
            size=16,
            symbol="star",
            line=dict(color="black", width=1),
        ),
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Value",
    legend_title="Legend",
    template="plotly_white",
    height=500,
)

st.plotly_chart(fig, use_container_width=True)

# --- DETAILED ANALYSIS AND ANOMALY TABLE ---
with st.expander("Detailed model description and anomaly table"):
    st.header("How to interpret the models")
    st.markdown(
        """
This dashboard combines three complementary anomaly detection methods:

- **Isolation Forest (red markers):** detects systemic anomalies in the *joint behaviour* of GDP, credit and debt.  
- **STL decomposition (green markers):** flags turning points and unusual movements within *each individual series*.  
- **Prophet (purple markers, based on GDP):** identifies values that deviate significantly from a model-based GDP forecast.  
- **Consensus (gold stars):** dates flagged by multiple models, corresponding to the strongest and most robust anomalies.
        """
    )

    st.header("Anomaly summary table")
    df_anomalies = df_final[df_final["anomaly_count"] > 0].copy()

    table_columns = [
        "gdp",
        "corporate_credit",
        "household_credit",
        "total_debt",
        "anomaly_isoforest",
        "anomaly_stl",
        "anomaly_prophet",
        "anomaly_count",
    ]

    # Sort by highest consensus first, then by date
    df_anomalies = (
        df_anomalies.sort_values("anomaly_count", ascending=False).sort_index()
    )

    st.dataframe(df_anomalies[table_columns])
