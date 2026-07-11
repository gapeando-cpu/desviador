import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, datetime

st.set_page_config(page_title="Comparador MSCI World vs Fondos", layout="wide")
st.title("Comparador MSCI World vs fondos indexados")

st.sidebar.header("Configuración")
start_date = st.sidebar.date_input("Fecha de inicio", value=date(2020, 1, 1))
end_date = st.sidebar.date_input("Fecha de fin", value=datetime.today().date())

benchmark_name = "MSCI World (proxy ETF)"
benchmark_ticker = st.sidebar.text_input("Ticker benchmark", value="URTH")

fund_inputs = {
    "Vanguard Global Stock Index Fund": st.sidebar.text_input("Ticker Vanguard", value=""),
    "iShares Developed World Index Fund": st.sidebar.text_input("Ticker iShares", value=""),
    "Fidelity MSCI World Index Fund": st.sidebar.text_input("Ticker Fidelity", value=""),
}

@st.cache_data(ttl=3600)
def download_series(ticker, start, end):
    if not ticker:
        return None
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.dropna()
        if len(s) < 10:
            return None
        s.index = pd.to_datetime(s.index)
        return s
    except Exception:
        return None


def annualized_return(series):
    days = (series.index[-1] - series.index[0]).days
    if days < 30:
        return np.nan
    years = days / 365.25
    return ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100


def tracking_error(active_returns):
    active_returns = active_returns.dropna()
    if active_returns.empty:
        return np.nan
    return active_returns.std() * np.sqrt(252) * 100


def max_drawdown(series):
    dd = series / series.cummax() - 1
    return dd.min() * 100


if start_date >= end_date:
    st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
    st.stop()

benchmark = download_series(benchmark_ticker, start_date, end_date)
if benchmark is None:
    st.error("No se pudo descargar el benchmark. Revisa el ticker.")
    st.stop()

series = {benchmark_name: benchmark}
failed = []

for name, ticker in fund_inputs.items():
    if ticker.strip():
        s = download_series(ticker.strip(), start_date, end_date)
        if s is not None:
            series[name] = s
        else:
            failed.append(f"{name}: {ticker}")

if len(series) < 2:
    st.warning("Introduce al menos un ticker de fondo válido además del benchmark.")
    st.stop()

if failed:
    with st.expander("Tickers no cargados"):
        st.text("\n".join(failed))

data = pd.concat(series, axis=1).sort_index().ffill().dropna()
if len(data) < 10:
    st.error("No hay suficiente histórico común para comparar.")
    st.stop()

norm = data / data.iloc[0] * 100
benchmark_norm = norm[benchmark_name]

fund_cols = [c for c in norm.columns if c != benchmark_name]
gap = norm[fund_cols].div(benchmark_norm, axis=0) - 1

rets = data.pct_change().dropna()
active_rets = rets[fund_cols].sub(rets[benchmark_name], axis=0)

fig1 = go.Figure()
for col in norm.columns:
    fig1.add_trace(go.Scatter(x=norm.index, y=norm[col], mode="lines", name=col))
fig1.update_layout(template="plotly_white", height=520, yaxis_title="Base 100", xaxis_title="Fecha")
st.subheader("Evolución base 100")
st.plotly_chart(fig1, use_container_width=True)

fig2 = go.Figure()
for col in gap.columns:
    fig2.add_trace(go.Scatter(x=gap.index, y=gap[col] * 100, mode="lines", name=col))
fig2.add_hline(y=0, line_dash="dash", line_color="gray")
fig2.update_layout(template="plotly_white", height=420, yaxis_title="Diferencia vs benchmark (%)", xaxis_title="Fecha")
st.subheader("Separación frente al benchmark")
st.plotly_chart(fig2, use_container_width=True)

summary_rows = []
for col in fund_cols:
    summary_rows.append(
        {
            "Fondo": col,
            "Rentabilidad total fondo (%)": round((norm[col].iloc[-1] / 100 - 1) * 100, 2),
            "Rentabilidad benchmark (%)": round((benchmark_norm.iloc[-1] / 100 - 1) * 100, 2),
            "Tracking difference final (%)": round((norm[col].iloc[-1] / benchmark_norm.iloc[-1] - 1) * 100, 2),
            "Tracking error anualizado (%)": round(tracking_error(active_rets[col]), 2),
            "Máx drawdown fondo (%)": round(max_drawdown(norm[col]), 2),
            "Máx separación histórica (%)": round(gap[col].abs().max() * 100, 2),
            "Rentabilidad anualizada (%)": round(annualized_return(norm[col]), 2),
        }
    )

summary = pd.DataFrame(summary_rows)
st.subheader("Resumen comparativo")
st.dataframe(summary, use_container_width=True)

csv_data = summary.to_csv(index=False).encode("utf-8")
st.download_button(
    "Descargar resumen CSV",
    data=csv_data,
    file_name="resumen_comparador_msci_world.csv",
    mime="text/csv",
)
