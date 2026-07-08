import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime

st.set_page_config(page_title="Comparador MSCI World vs Fondos", layout="wide")
st.title("🌍 Comparador MSCI World vs Fondos indexados")

st.sidebar.header("Configuración")
start_date = st.sidebar.date_input("Fecha de inicio", date(2020, 1, 1))
end_date = st.sidebar.date_input("Fecha de fin", datetime.today().date())

# Benchmark proxy: ETF MSCI World
benchmark_name = "MSCI World (proxy ETF)"
benchmark_ticker = "URTH"

funds = {
    "Vanguard Global Stock Index EUR Acc": "0P0000TKZO.F",   # ejemplo, revisar ticker real disponible
    "iShares Developed World Index Fund": "0P0000YWLZ.F",    # ejemplo, revisar ticker real disponible
    "Fidelity MSCI World Index Fund": "0P0001...? ",         # revisar ticker real disponible
}

@st.cache_data(ttl=3600)
def download_series(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return None
    if "Close" in df.columns:
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.dropna()
    return None

series = {}

benchmark = download_series(benchmark_ticker, start_date, end_date)
if benchmark is None:
    st.error("No se pudo descargar el benchmark.")
    st.stop()

series[benchmark_name] = benchmark

for name, ticker in funds.items():
    s = download_series(ticker, start_date, end_date)
    if s is not None and len(s) > 10:
        series[name] = s
    else:
        st.warning(f"No se pudieron cargar datos para {name}")

if len(series) < 2:
    st.error("No hay suficientes series para comparar.")
    st.stop()

data = pd.concat(series, axis=1).dropna().sort_index()

# Normalización base 100
norm = data / data.iloc[0] * 100

# Separación frente al benchmark
gap = norm.div(norm[benchmark_name], axis=0) - 1
gap = gap.drop(columns=[benchmark_name])

# Rentabilidades diarias
rets = data.pct_change().dropna()
active_rets = rets.drop(columns=[benchmark_name]).sub(rets[benchmark_name], axis=0)

def tracking_error(active_returns):
    return active_returns.std() * (252 ** 0.5) * 100

def tracking_difference(norm_series, benchmark_series):
    return (norm_series.iloc[-1] / benchmark_series.iloc[-1] - 1) * 100

def max_relative_gap(gap_series):
    return gap_series.abs().max() * 100

# Gráfico evolución normalizada
st.subheader("Evolución base 100")
fig1 = go.Figure()
for col in norm.columns:
    fig1.add_trace(go.Scatter(x=norm.index, y=norm[col], mode="lines", name=col))
fig1.update_layout(template="plotly_white", height=550, yaxis_title="Base 100")
st.plotly_chart(fig1, use_container_width=True)

# Gráfico separación frente al benchmark
st.subheader("Separación frente al benchmark")
fig2 = go.Figure()
for col in gap.columns:
    fig2.add_trace(go.Scatter(x=gap.index, y=gap[col] * 100, mode="lines", name=col))
fig2.add_hline(y=0, line_dash="dash", line_color="gray")
fig2.update_layout(template="plotly_white", height=500, yaxis_title="Diferencia vs benchmark (%)")
st.plotly_chart(fig2, use_container_width=True)

# Resumen
summary = {}
for col in gap.columns:
    summary[col] = {
        "Rent. acumulada fondo (%)": round((norm[col].iloc[-1] / 100 - 1) * 100, 2),
        "Rent. acumulada benchmark (%)": round((norm[benchmark_name].iloc[-1] / 100 - 1) * 100, 2),
        "Tracking difference final (%)": round(tracking_difference(norm[col], norm[benchmark_name]), 2),
        "Tracking error anualizado (%)": round(tracking_error(active_rets[col]), 2),
        "Máx. separación histórica (%)": round(max_relative_gap(gap[col]), 2),
    }

summary_df = pd.DataFrame(summary).T
st.subheader("Resumen comparativo")
st.dataframe(summary_df)

st.caption("Nota: usa tickers/series realmente disponibles en Yahoo Finance o sustituye la fuente por otra API de fondos.")
