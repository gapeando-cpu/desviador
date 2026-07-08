import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime

st.set_page_config(page_title="Comparador MSCI World vs Fondos indexados", layout="wide")
st.title("🌍 Comparador MSCI World vs Fondos indexados")

st.sidebar.header("Configuración")

# Fechas
if "start_date" not in st.session_state:
    st.session_state.start_date = date(2020, 1, 1)
if "end_date" not in st.session_state:
    st.session_state.end_date = datetime.today().date()

start_date = st.sidebar.date_input("Fecha de inicio", st.session_state.start_date, key="start_date")
end_date = st.sidebar.date_input("Fecha de fin", st.session_state.end_date, key="end_date")

# Benchmark MSCI World por proxy ETF, solo como referencia
# Se mantiene para medir separación, pero NO se muestra como activo "invertible" comparado.
benchmark_candidates = {
    "MSCI World proxy - URTH": "URTH",
    "MSCI World proxy - IWDA.AS": "IWDA.AS",
    "MSCI World proxy - SWDA.L": "SWDA.L",
}

# Fondos solicitados
# Ojo: en Yahoo Finance los fondos suelen usar sufijos de mercado.
# Vanguard y Fidelity tienen referencias públicas localizadas.
# El iShares puede requerir ajustar el sufijo si Yahoo no lo resuelve.
funds = {
    "iShares Developed World Index Fund | IE000ZYRH0Q7 | TER 0,06%": [
        "IE000ZYRH0Q7",
        "IE000ZYRH0Q7.IR",
        "IE000ZYRH0Q7.SG",
        "IE000ZYRH0Q7.F"
    ],
    "Fidelity MSCI World Index Fund | IE00BYX5NX33 | TER 0,12%": [
        "IE00BYX5NX33.SG",
        "IE00BYX5NX33"
    ],
    "Vanguard Global Stock Index Fund | IE00B03HD191 | TER 0,18%": [
        "IE00B03HD191.IR",
        "IE00B03HD191"
    ],
}

st.sidebar.subheader("Fondos a comparar")
selected_funds = []
for name in funds:
    if st.sidebar.checkbox(name, value=True, key=f"chk_{name}"):
        selected_funds.append(name)

if not selected_funds:
    st.warning("Selecciona al menos un fondo para comparar.")
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


def find_first_working_ticker(candidates, start, end):
    for ticker in candidates:
        s = download_series(ticker, start, end)
        if s is not None and len(s) > 10:
            return ticker, s
    return None, None


def find_benchmark(start, end):
    for name, ticker in benchmark_candidates.items():
        s = download_series(ticker, start, end)
        if s is not None and len(s) > 10:
            return name, ticker, s
    return None, None, None


def tracking_error(active_returns):
    active_returns = active_returns.dropna()
    if active_returns.empty:
        return float("nan")
    return active_returns.std() * (252 ** 0.5) * 100


def tracking_difference(norm_series, benchmark_series):
    return (norm_series.iloc[-1] / benchmark_series.iloc[-1] - 1) * 100


def max_relative_gap(gap_series):
    gap_series = gap_series.dropna()
    if gap_series.empty:
        return float("nan")
    return gap_series.abs().max() * 100


def annualized_return(price_series):
    days = (price_series.index[-1] - price_series.index[0]).days
    if days < 30:
        return float("nan")
    years = days / 365.25
    return ((price_series.iloc[-1] / price_series.iloc[0]) ** (1 / years) - 1) * 100


# Benchmark
benchmark_name, benchmark_ticker, benchmark = find_benchmark(start_date, end_date)

if benchmark is None:
    st.error("No se pudo descargar ningún benchmark MSCI World.")
    st.info("Prueba con otro rango de fechas o revisa si Yahoo Finance responde en este momento.")
    st.stop()

st.sidebar.success(f"Benchmark cargado: {benchmark_name} [{benchmark_ticker}]")

# Descargar fondos
series = {benchmark_name: benchmark}
resolved_tickers = {}
failed_funds = []

for fund_name in selected_funds:
    candidates = funds[fund_name]
    resolved_ticker, s = find_first_working_ticker(candidates, start_date, end_date)

    if s is not None:
        series[fund_name] = s
        resolved_tickers[fund_name] = resolved_ticker
        st.sidebar.success(f"✅ {fund_name}")
        st.sidebar.caption(f"Ticker usado: {resolved_ticker}")
    else:
        failed_funds.append(fund_name)
        st.sidebar.warning(f"⚠️ Sin datos: {fund_name}")

if len(series) < 2:
    st.error("No hay suficientes fondos con datos para comparar contra el benchmark.")
    st.stop()

# Unir y alinear
data = pd.concat(series, axis=1)
data = data.dropna(how="all").sort_index()
data = data.ffill().dropna()

if data.shape[0] < 10 or data.shape[1] < 2:
    st.error("Tras alinear fechas, no quedan suficientes datos comparables.")
    st.stop()

# Normalización base 100
norm = data / data.iloc[0] * 100

# Separación relativa frente al benchmark
gap = norm.div(norm[benchmark_name], axis=0) - 1
gap = gap.drop(columns=[benchmark_name], errors="ignore")

# Retornos diarios
rets = data.pct_change().dropna()
active_rets = rets.drop(columns=[benchmark_name], errors="ignore").sub(rets[benchmark_name], axis=0)

# Información superior
st.subheader("Referencia utilizada")
st.write(f"**Benchmark activo:** {benchmark_name} (`{benchmark_ticker}`)")

if resolved_tickers:
    st.subheader("Tickers resueltos")
    resolved_df = pd.DataFrame(
        [{"Fondo": k, "Ticker usado": v} for k, v in resolved_tickers.items()]
    )
    st.dataframe(resolved_df, use_container_width=True)

if failed_funds:
    st.warning("Fondos sin datos en Yahoo Finance para este rango:")
    for f in failed_funds:
        st.write(f"- {f}")

# Gráfico base 100
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

# Gráfico de gap
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

# Resumen comparativo
summary_data = {}
for col in gap.columns:
    serie = data[col]

    summary_data[col] = {
        "Ticker usado": resolved_tickers.get(col, "—"),
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

# Última desviación
st.subheader("Desviación actual frente al benchmark")
last_gap = (gap.iloc[-1] * 100).sort_values()
last_gap_df = pd.DataFrame({"Gap actual vs benchmark (%)": last_gap.round(3)})
st.dataframe(last_gap_df, use_container_width=True)

st.caption(
    "Nota: la comparación se hace contra un proxy del MSCI World descargado desde Yahoo Finance. "
    "En fondos de inversión, Yahoo no siempre publica todas las clases o mercados con el mismo sufijo; "
    "por eso esta app intenta varias alternativas antes de dar error."
)