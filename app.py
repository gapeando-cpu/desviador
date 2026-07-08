import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime

st.set_page_config(page_title="Comparador MSCI World vs Fondos", layout="wide")
st.title("🌍 Comparador MSCI World vs Fondos indexados")

st.sidebar.header("Configuración")

# Fechas
if "start_date" not in st.session_state:
    st.session_state.start_date = date(2020, 1, 1)
if "end_date" not in st.session_state:
    st.session_state.end_date = datetime.today().date()

start_date = st.sidebar.date_input("Fecha de inicio", st.session_state.start_date, key="start_date")
end_date = st.sidebar.date_input("Fecha de fin", st.session_state.end_date, key="end_date")

# Benchmark candidates (fallback)
benchmark_candidates = {
    "iShares MSCI World ETF (URTH)": "URTH",
    "iShares Core MSCI World UCITS ETF (IWDA.AS)": "IWDA.AS",
    "iShares Core MSCI World UCITS ETF (SWDA.L)": "SWDA.L",
}

# Series a comparar
# Nota: los fondos europeos muchas veces no descargan bien en Yahoo;
# por eso aquí puedes mezclar fondos/ETFs según disponibilidad real.
comparison_assets = {
    "Vanguard FTSE Developed World UCITS ETF (VEVE.L)": "VEVE.L",
    "iShares Core MSCI World UCITS ETF (IWDA.AS)": "IWDA.AS",
    "iShares Core MSCI World UCITS ETF (SWDA.L)": "SWDA.L",
    "Amundi MSCI World UCITS ETF (CW8.PA)": "CW8.PA",
}

st.sidebar.subheader("Activos a comparar")
selected_assets = []
for name in comparison_assets:
    if st.sidebar.checkbox(name, value=True, key=f"chk_{name}"):
        selected_assets.append(name)

if not selected_assets:
    st.warning("Selecciona al menos un activo para comparar.")
    st.stop()


@st.cache_data(ttl=3600)
def download_series(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None

        if "Close" not in df.columns:
            return None

        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = close.dropna()
        if len(close) < 10:
            return None

        return close
    except Exception:
        return None


def find_benchmark(start, end):
    for name, ticker in benchmark_candidates.items():
        s = download_series(ticker, start, end)
        if s is not None and len(s) > 10:
            return name, ticker, s
    return None, None, None


def tracking_error(active_returns):
    if active_returns.dropna().empty:
        return float("nan")
    return active_returns.std() * (252 ** 0.5) * 100


def tracking_difference(norm_series, benchmark_series):
    return (norm_series.iloc[-1] / benchmark_series.iloc[-1] - 1) * 100


def max_relative_gap(gap_series):
    if gap_series.dropna().empty:
        return float("nan")
    return gap_series.abs().max() * 100


def annualized_return(price_series):
    days = (price_series.index[-1] - price_series.index[0]).days
    if days < 30:
        return float("nan")
    years = days / 365.25
    return ((price_series.iloc[-1] / price_series.iloc[0]) ** (1 / years) - 1) * 100


# Descargar benchmark con fallback
benchmark_name, benchmark_ticker, benchmark = find_benchmark(start_date, end_date)

if benchmark is None:
    st.error("No se pudo descargar ningún benchmark.")
    st.info("Prueba con otro rango de fechas o revisa la conectividad con Yahoo Finance.")
    st.stop()

st.sidebar.success(f"Benchmark cargado: {benchmark_name} [{benchmark_ticker}]")

# Descargar activos
series = {benchmark_name: benchmark}
failed_assets = []

for name in selected_assets:
    ticker = comparison_assets[name]
    s = download_series(ticker, start_date, end_date)
    if s is not None and len(s) > 10:
        series[name] = s
        st.sidebar.success(f"✅ {name}")
    else:
        failed_assets.append(name)
        st.sidebar.warning(f"⚠️ Sin datos: {name}")

if len(series) < 2:
    st.error("No hay suficientes series para comparar contra el benchmark.")
    st.stop()

# Unir y limpiar
data = pd.concat(series, axis=1)
data = data.dropna(how="all").sort_index()
data = data.ffill().dropna()

if data.shape[0] < 10 or data.shape[1] < 2:
    st.error("Tras alinear fechas, no quedan suficientes datos comparables.")
    st.stop()

# Base 100
norm = data / data.iloc[0] * 100

# Gap relativo frente al benchmark
gap = norm.div(norm[benchmark_name], axis=0) - 1
gap = gap.drop(columns=[benchmark_name], errors="ignore")

# Retornos diarios
rets = data.pct_change().dropna()
active_rets = rets.drop(columns=[benchmark_name], errors="ignore").sub(rets[benchmark_name], axis=0)

# Info superior
st.subheader("Referencia usada")
st.write(f"**Benchmark activo:** {benchmark_name} (`{benchmark_ticker}`)")

if failed_assets:
    st.caption("Activos sin datos en Yahoo Finance para este rango: " + ", ".join(failed_assets))

# Gráfico 1: evolución base 100
st.subheader(f"Evolución base 100 ({start_date} a {end_date})")
fig1 = go.Figure()

for col in norm.columns:
    width = 3 if col == benchmark_name else 2
    fig1.add_trace(
        go.Scatter(
            x=norm.index,
            y=norm[col],
            mode="lines",
            name=col,
            line=dict(width=width)
        )
    )

fig1.update_layout(
    height=600,
    template="plotly_white",
    xaxis_title="Fecha",
    yaxis_title="Base 100"
)
st.plotly_chart(fig1, use_container_width=True)

# Gráfico 2: separación frente al benchmark
st.subheader("Separación frente al benchmark")
fig2 = go.Figure()

for col in gap.columns:
    fig2.add_trace(
        go.Scatter(
            x=gap.index,
            y=gap[col] * 100,
            mode="lines",
            name=col
        )
    )

fig2.add_hline(y=0, line_dash="dash", line_color="gray")

fig2.update_layout(
    height=500,
    template="plotly_white",
    xaxis_title="Fecha",
    yaxis_title="Diferencia vs benchmark (%)"
)
st.plotly_chart(fig2, use_container_width=True)

# Resumen
summary_data = {}
for col in gap.columns:
    serie = data[col]
    bench = data[benchmark_name]

    summary_data[col] = {
        "Rentabilidad total fondo (%)": round((norm[col].iloc[-1] / 100 - 1) * 100, 2),
        "Rentabilidad total benchmark (%)": round((norm[benchmark_name].iloc[-1] / 100 - 1) * 100, 2),
        "Rentabilidad anualizada fondo (%)": round(annualized_return(serie), 2),
        "Tracking difference final (%)": round(tracking_difference(norm[col], norm[benchmark_name]), 2),
        "Tracking error anualizado (%)": round(tracking_error(active_rets[col]), 2) if col in active_rets.columns else None,
        "Máx. separación histórica (%)": round(max_relative_gap(gap[col]), 2),
    }

summary = pd.DataFrame(summary_data).T

st.subheader("Resumen comparativo")
st.dataframe(summary, use_container_width=True)

# Tabla extra: últimos gaps
st.subheader("Última desviación observada")
last_gap = (gap.iloc[-1] * 100).sort_values()
st.dataframe(
    pd.DataFrame({"Gap actual vs benchmark (%)": last_gap.round(3)}),
    use_container_width=True
)

st.caption(
    "Nota: esta app usa Yahoo Finance vía yfinance. Para fondos de inversión indexados tradicionales, "
    "muchas clases no están bien cubiertas; los ETF suelen funcionar mejor como proxy."
)