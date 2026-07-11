import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import date, datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Comparador MSCI World vs Fondos", layout="wide")

FUNDS = {
    "iShares Developed World Index": {
        "isin": "IE000ZYRH0Q7",
        "ter": 0.06,
        "yahoo_candidates": ["IE000ZYRH0Q7", "IE000ZYRH0Q7.IR", "IE000ZYRH0Q7.SG", "IE000ZYRH0Q7.F"],
        "stooq_candidates": []
    },
    "Fidelity MSCI World Index Fund": {
        "isin": "IE00BYX5NX33",
        "ter": 0.12,
        "yahoo_candidates": ["IE00BYX5NX33.SG", "IE00BYX5NX33"],
        "stooq_candidates": []
    },
    "Vanguard Global Stock Index Fund": {
        "isin": "IE00B03HD191",
        "ter": 0.18,
        "yahoo_candidates": ["IE00B03HD191.IR", "IE00B03HD191"],
        "stooq_candidates": []
    },
}

BENCHMARKS = {
    "MSCI World (Ã­ndice Stooq)": {
        "stooq_candidates": ["r2.f"]
    },
    "MSCI World (ETF proxy Yahoo)": {
        "yahoo_candidates": ["URTH", "IWDA.AS", "SWDA.L"],
        "stooq_candidates": ["iwda.uk"]
    }
}


def card_style(text, border="#1f2c45"):
    return f"""
    <div style='background:linear-gradient(180deg,#101827,#0c1322);border:1px solid {border};
    border-radius:14px;padding:14px 16px;height:100%;box-shadow:0 8px 24px rgba(0,0,0,.22)'>{text}</div>
    """


@st.cache_data(ttl=3600)
def download_yahoo_series(ticker, start, end):
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
        s.name = ticker
        return s
    except Exception:
        return None


@st.cache_data(ttl=3600)
def download_stooq_series(symbol, start, end):
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        txt = r.text.strip()
        if not txt or "No data" in txt:
            return None
        df = pd.read_csv(StringIO(txt))
        if df.empty or "Date" not in df.columns:
            return None
        close_col = "Close" if "Close" in df.columns else df.columns[-1]
        s = pd.Series(df[close_col].values, index=pd.to_datetime(df["Date"]), name=symbol).sort_index()
        s = s[(s.index >= pd.to_datetime(start)) & (s.index <= pd.to_datetime(end))].dropna()
        if len(s) < 10:
            return None
        return s
    except Exception:
        return None


@st.cache_data(ttl=3600)
def resolve_series(name, meta, start, end):
    tried = []
    for ticker in meta.get("yahoo_candidates", []):
        tried.append(f"Yahoo:{ticker}")
        s = download_yahoo_series(ticker, start, end)
        if s is not None:
            return {"name": name, "source": "Yahoo Finance", "ticker": ticker, "series": s}
    for symbol in meta.get("stooq_candidates", []):
        tried.append(f"Stooq:{symbol}")
        s = download_stooq_series(symbol, start, end)
        if s is not None:
            return {"name": name, "source": "Stooq", "ticker": symbol, "series": s}
    return {"name": name, "source": None, "ticker": None, "series": None, "tried": tried}



def annualized_return(series):
    days = (series.index[-1] - series.index[0]).days
    if days < 30:
        return np.nan
    years = days / 365.25
    return ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100



def max_drawdown(series):
    roll_max = series.cummax()
    dd = series / roll_max - 1
    return dd.min() * 100



def tracking_error(active_returns):
    ar = active_returns.dropna()
    if ar.empty:
        return np.nan
    return ar.std() * np.sqrt(252) * 100



def downside_capture(fund_returns, benchmark_returns):
    mask = benchmark_returns < 0
    if mask.sum() < 3:
        return np.nan
    bench = benchmark_returns[mask]
    fund = fund_returns[mask]
    if bench.mean() == 0:
        return np.nan
    return (fund.mean() / bench.mean()) * 100



def make_line_chart(df_norm, benchmark_name):
    fig = go.Figure()
    colors = {
        benchmark_name: "#9fb4d9",
        "iShares Developed World Index": "#f0a128",
        "Fidelity MSCI World Index Fund": "#38d1f4",
        "Vanguard Global Stock Index Fund": "#ae8cff",
    }
    for col in df_norm.columns:
        fig.add_trace(go.Scatter(
            x=df_norm.index,
            y=df_norm[col],
            mode="lines",
            name=col,
            line=dict(color=colors.get(col, "#d9e4ff"), width=3 if col == benchmark_name else 2.3),
            fill="tozeroy" if col == benchmark_name else None,
            fillcolor="rgba(159,180,217,0.05)" if col == benchmark_name else None,
        ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,16,32,0.55)",
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.08),
        xaxis_title="",
        yaxis_title="Base 100",
    )
    fig.update_xaxes(gridcolor="rgba(120,155,220,.08)")
    fig.update_yaxes(gridcolor="rgba(120,155,220,.08)")
    return fig



def make_gap_chart(gap_df):
    fig = go.Figure()
    colors = {
        "iShares Developed World Index": "#f0a128",
        "Fidelity MSCI World Index Fund": "#38d1f4",
        "Vanguard Global Stock Index Fund": "#ae8cff",
    }
    for col in gap_df.columns:
        fig.add_trace(go.Scatter(
            x=gap_df.index,
            y=gap_df[col] * 100,
            mode="lines",
            name=col,
            line=dict(color=colors.get(col, "#d9e4ff"), width=2.2)
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(220,230,255,.4)")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,16,32,0.55)",
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.08),
        yaxis_title="Gap vs benchmark (%)",
        xaxis_title="",
    )
    fig.update_xaxes(gridcolor="rgba(120,155,220,.08)")
    fig.update_yaxes(gridcolor="rgba(120,155,220,.08)")
    return fig



def make_drawdown_chart(drawdown_df):
    fig = go.Figure()
    colors = {
        "iShares Developed World Index": "#f0a128",
        "Fidelity MSCI World Index Fund": "#38d1f4",
        "Vanguard Global Stock Index Fund": "#ae8cff",
    }
    for col in drawdown_df.columns:
        fig.add_trace(go.Scatter(
            x=drawdown_df.index,
            y=drawdown_df[col],
            mode="lines",
            name=col,
            line=dict(color=colors.get(col, "#d9e4ff"), width=2.2),
            fill="tozeroy",
            fillcolor="rgba(180,190,255,.03)"
        ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,16,32,0.55)",
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.08),
        yaxis_title="Drawdown (%)",
        xaxis_title="",
    )
    fig.update_xaxes(gridcolor="rgba(120,155,220,.08)")
    fig.update_yaxes(gridcolor="rgba(120,155,220,.08)")
    return fig



def make_bar_chart(summary, order):
    periods = ["Rent. total (%)", "Rent. anualizada (%)", "Tracking diff final (%)"]
    colors = ["#f0a128", "#38d1f4", "#ae8cff"]
    fig = go.Figure()
    for i, fund in enumerate(order):
        vals = [summary.loc[fund, c] for c in periods]
        fig.add_trace(go.Bar(name=fund, x=["Total", "Anualizada", "TrackDiff"], y=vals, marker_color=colors[i]))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,16,32,0.55)",
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        barmode="group",
        legend=dict(orientation="h", y=1.08),
        yaxis_title="%",
        xaxis_title="",
    )
    fig.update_xaxes(gridcolor="rgba(120,155,220,.08)")
    fig.update_yaxes(gridcolor="rgba(120,155,220,.08)")
    return fig


st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top right, rgba(58,93,171,.12), transparent 28%), linear-gradient(180deg, #05070d 0%, #070b14 35%, #050811 100%); color: #ecf2ff; }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1380px; }
h1,h2,h3 { letter-spacing: -.02em; }
div[data-testid="stMetric"] { background: linear-gradient(180deg,#101827,#0c1322); border:1px solid #1f2c45; border-radius: 14px; padding: 8px 10px; }
div[data-testid="stDataFrame"] { border:1px solid #1f2c45; border-radius:14px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

st.title("ðŸŒ Comparador MSCI World vs fondos indexados")
st.caption("Compara cuÃ¡nto se separan del benchmark los fondos que replican bolsa global desarrollada con datos reales de Yahoo Finance y fallback a Stooq cuando haga falta.")

colA, colB, colC = st.columns([1.2, 1.2, 1.4])
with colA:
    start_date = st.date_input("Fecha inicio", value=date(2021, 7, 1))
with colB:
    end_date = st.date_input("Fecha fin", value=datetime.today().date())
with colC:
    benchmark_label = st.selectbox("Benchmark", list(BENCHMARKS.keys()), index=0)

if start_date >= end_date:
    st.error("La fecha de inicio debe ser anterior a la fecha final.")
    st.stop()

benchmark_result = resolve_series(benchmark_label, BENCHMARKS[benchmark_label], start_date, end_date)
if benchmark_result["series"] is None:
    st.error("No se pudo descargar el benchmark. Prueba con otro rango o usa el benchmark proxy ETF.")
    st.write(benchmark_result.get("tried", []))
    st.stop()

benchmark_name = benchmark_result["name"]
benchmark_series = benchmark_result["series"]

loaded = {}
failed = []
for fund_name, meta in FUNDS.items():
    res = resolve_series(fund_name, meta, start_date, end_date)
    if res["series"] is not None:
        loaded[fund_name] = res
    else:
        failed.append((fund_name, res.get("tried", [])))

if not loaded:
    st.error("No se pudo descargar ninguno de los fondos en el rango seleccionado.")
    st.stop()

all_series = {benchmark_name: benchmark_series}
for fund_name, res in loaded.items():
    all_series[fund_name] = res["series"]

prices = pd.concat(all_series, axis=1).dropna(how="all").sort_index().ffill().dropna()
if prices.shape[0] < 30:
    st.error("No hay suficientes observaciones comunes tras alinear fechas.")
    st.stop()

norm = prices / prices.iloc[0] * 100
fund_cols = [c for c in prices.columns if c != benchmark_name]
rets = prices.pct_change().dropna()
gap = norm[fund_cols].div(norm[benchmark_name], axis=0) - 1
active_rets = rets[fund_cols].sub(rets[benchmark_name], axis=0)
drawdowns = prices[fund_cols].div(prices[fund_cols].cummax()).sub(1).mul(100)

summary_rows = []
for fund in fund_cols:
    meta = FUNDS[fund]
    serie = prices[fund]
    td_final = (norm[fund].iloc[-1] / norm[benchmark_name].iloc[-1] - 1) * 100
    row = {
        "Fondo": fund,
        "ISIN": meta["isin"],
        "TER (%)": meta["ter"],
        "Ticker usado": loaded[fund]["ticker"],
        "Fuente": loaded[fund]["source"],
        "Rent. total (%)": round((norm[fund].iloc[-1] / 100 - 1) * 100, 2),
        "Rent. anualizada (%)": round(annualized_return(serie), 2),
        "Tracking diff final (%)": round(td_final, 2),
        "Tracking error (%)": round(tracking_error(active_rets[fund]), 2),
        "MÃ¡x drawdown (%)": round(max_drawdown(serie), 2),
        "Downside capture (%)": round(downside_capture(rets[fund], rets[benchmark_name]), 2),
        "Gap mÃ¡ximo abs (%)": round(gap[fund].abs().max() * 100, 2),
    }
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows).set_index("Fondo").sort_values("Tracking diff final (%)", ascending=False)

best_return = summary["Rent. total (%)"].idxmax()
lowest_dd = summary["MÃ¡x drawdown (%)"].idxmax()
closest_index = summary["Tracking diff final (%)"].abs().idxmin()
cheapest = summary["TER (%)"].idxmin()

st.markdown(card_style(
    f"<div style='font-size:14px'><span style='color:#f7c15f;font-weight:800'>Veredicto.</span> "
    f"Por rentabilidad gana <b>{best_return}</b>; el que menos cae es <b>{lowest_dd}</b>; "
    f"el mÃ¡s pegado a la referencia es <b>{closest_index}</b>; el mÃ¡s barato es <b>{cheapest}</b>.</div>",
    border="rgba(240,161,40,.55)"
), unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Benchmark usado", benchmark_result['ticker'])
m2.metric("Fuente benchmark", benchmark_result['source'])
m3.metric("Observaciones comunes", f"{len(prices):,}".replace(",", "."))
m4.metric("Fondos con datos", str(len(fund_cols)))

if failed:
    with st.expander("Fondos no descargados"):
        for name, tried in failed:
            st.write(f"**{name}**")
            st.code("\n".join(tried) if tried else "Sin candidatos configurados")

c1, c2 = st.columns([1.1, 2.2])
with c1:
    st.markdown(card_style(
        "<div style='font-size:11px;color:#9db0d4;text-transform:uppercase;letter-spacing:.08em;font-weight:700;margin-bottom:8px'>Fondos cargados</div>"
        + "".join([
            f"<div style='display:flex;justify-content:space-between;gap:10px;padding:8px 0;border-top:1px solid rgba(135,163,216,.08);font-size:12px'>"
            f"<span>{fund}</span><span style='color:#7fa0d8'>{loaded[fund]['ticker']}</span></div>" for fund in fund_cols
        ])
    ), unsafe_allow_html=True)

    st.dataframe(summary[["ISIN", "TER (%)", "Ticker usado", "Fuente"]], use_container_width=True)

with c2:
    st.subheader("Crecimiento de 100")
    st.plotly_chart(make_line_chart(norm[[benchmark_name] + fund_cols], benchmark_name), use_container_width=True)

st.subheader("SeparaciÃ³n frente al benchmark")
st.plotly_chart(make_gap_chart(gap[fund_cols]), use_container_width=True)

c3, c4 = st.columns([1.4, 1.2])
with c3:
    st.subheader("Drawdown")
    st.plotly_chart(make_drawdown_chart(drawdowns[fund_cols]), use_container_width=True)
with c4:
    st.subheader("Resumen por fondo")
    st.dataframe(summary[[
        "Rent. total (%)", "Rent. anualizada (%)", "Tracking diff final (%)",
        "Tracking error (%)", "MÃ¡x drawdown (%)", "Gap mÃ¡ximo abs (%)"
    ]], use_container_width=True)

st.subheader("Comparativa rÃ¡pida")
st.plotly_chart(make_bar_chart(summary, fund_cols), use_container_width=True)

st.caption(
    f"Benchmark seleccionado: {benchmark_name} vÃ­a {benchmark_result['source']} ({benchmark_result['ticker']}). "
    f"Stooq ofrece histÃ³rico gratuito de sÃ­mbolos como IWDA.UK y del futuro/Ã­ndice MSCI World en R2.F, y Yahoo Finance publica histÃ³rico para fondos y ETFs en muchos sÃ­mbolos compatibles con yfinance. "
    f"Los datos de fondos pueden fallar segÃºn la clase y el sufijo de mercado disponible."
)