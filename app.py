import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import date, datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Comparador de carteras â€” Fondos", layout="wide")

CATALOGO = {
    "MSCI World / Desarrollados": [
        {
            "nombre": "iShares Developed World Index",
            "isin": "IE000ZYRH0Q7",
            "ter": 0.06,
            "yahoo": ["IE000ZYRH0Q7", "IE000ZYRH0Q7.IR", "IE000ZYRH0Q7.SG", "IE000ZYRH0Q7.F"],
            "stooq": []
        },
        {
            "nombre": "Fidelity MSCI World Index Fund",
            "isin": "IE00BYX5NX33",
            "ter": 0.12,
            "yahoo": ["IE00BYX5NX33.SG", "IE00BYX5NX33"],
            "stooq": []
        },
        {
            "nombre": "Vanguard Global Stock Index Fund",
            "isin": "IE00B03HD191",
            "ter": 0.18,
            "yahoo": ["IE00B03HD191.IR", "IE00B03HD191"],
            "stooq": []
        },
    ],
    "Global / ACWI": [
        {
            "nombre": "MSCI World proxy ETF",
            "isin": "URTH",
            "ter": 0.24,
            "yahoo": ["URTH", "IWDA.AS", "SWDA.L"],
            "stooq": ["iwda.uk"]
        }
    ]
}

BENCHMARKS = {
    "MSCI World (Ã­ndice Stooq)": {"yahoo": [], "stooq": ["r2.f"]},
    "MSCI World (proxy ETF Yahoo)": {"yahoo": ["URTH", "IWDA.AS", "SWDA.L"], "stooq": ["iwda.uk"]},
}


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
def resolve_series(candidates_yahoo, candidates_stooq, start, end):
    tried = []
    for ticker in candidates_yahoo:
        tried.append(f"Yahoo:{ticker}")
        s = download_yahoo_series(ticker, start, end)
        if s is not None:
            return s, ticker, "Yahoo Finance", tried
    for ticker in candidates_stooq:
        tried.append(f"Stooq:{ticker}")
        s = download_stooq_series(ticker, start, end)
        if s is not None:
            return s, ticker, "Stooq", tried
    return None, None, None, tried



def annualized_return(series):
    days = (series.index[-1] - series.index[0]).days
    if days < 30:
        return np.nan
    years = days / 365.25
    return ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100



def max_drawdown(series):
    cummax = series.cummax()
    dd = series / cummax - 1
    return dd.min() * 100



def calc_portfolio_value(prices, weights):
    w = pd.Series(weights, index=prices.columns, dtype=float)
    w = w / w.sum()
    normalized = prices / prices.iloc[0]
    return (normalized * w).sum(axis=1) * 100



def calc_drawdown(series):
    return (series / series.cummax() - 1) * 100



def rolling_consistency(series_dict, window_months=36):
    monthly = pd.DataFrame(series_dict).resample("M").last().dropna()
    if len(monthly) <= window_months:
        return None
    wins = {c: 0 for c in monthly.columns}
    total = 0
    for i in range(len(monthly) - window_months):
        block = monthly.iloc[i:i + window_months + 1]
        perf = block.iloc[-1] / block.iloc[0] - 1
        winner = perf.idxmax()
        wins[winner] += 1
        total += 1
    return {k: (v / total * 100) for k, v in wins.items()}



def get_all_funds():
    out = []
    for grupo in CATALOGO.values():
        out.extend(grupo)
    return out


ALL_FUNDS = get_all_funds()
NAME_TO_META = {f['nombre']: f for f in ALL_FUNDS}

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top right, rgba(58,93,171,.12), transparent 28%), linear-gradient(180deg, #05070d 0%, #070b14 35%, #050811 100%);
    color: #ecf2ff;
}
.block-container {max-width: 1380px; padding-top: 1rem; padding-bottom: 2rem;}
h1,h2,h3 {letter-spacing: -.02em;}
.panel {
    background: linear-gradient(180deg, rgba(14,20,35,.98), rgba(9,14,26,.98));
    border: 1px solid #1b2943;
    border-radius: 14px;
    padding: 12px 14px;
    box-shadow: 0 12px 32px rgba(0,0,0,.28);
}
.verdict {
    border: 1px solid rgba(240,161,40,.55);
    border-radius: 11px;
    background: linear-gradient(180deg, rgba(34,24,7,.66), rgba(24,18,8,.72));
    color: #f6f0de;
    padding: 12px 14px;
    font-size: 14px;
}
div[data-testid="stMetric"] {
    background: linear-gradient(180deg,#101827,#0c1322);
    border:1px solid #1f2c45;
    border-radius: 14px;
    padding: 8px 10px;
}
div[data-testid="stDataFrame"] {
    border:1px solid #1f2c45;
    border-radius:14px;
    overflow:hidden;
}
.smallcap {font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:#9caecc; font-weight:700;}
.port-title-a {color:#f0a128; font-weight:800;}
.port-title-b {color:#38d1f4; font-weight:800;}
.port-title-c {color:#ae8cff; font-weight:800;}
</style>
""", unsafe_allow_html=True)

st.title("Comparador de carteras â€” Fondos")
st.caption("Mismo enfoque visual que la web de referencia, pero con selecciÃ³n manual de fondos, pesos y descarga de datos reales desde Yahoo Finance/Stooq.")

quick = st.container(border=False)
with quick:
    st.markdown("<div class='panel'><div class='smallcap'>Empieza rÃ¡pido</div><div style='font-size:12px;color:#90a3c8'>Configura tus carteras A, B y C con fondos reales, pesos personalizados y benchmark opcional.</div></div>", unsafe_allow_html=True)

c0, c1, c2 = st.columns([1,1,1])
with c0:
    start_date = st.date_input("Fecha inicio", value=date(2021, 7, 1), key="start")
with c1:
    end_date = st.date_input("Fecha fin", value=datetime.today().date(), key="end")
with c2:
    benchmark_label = st.selectbox("Benchmark", list(BENCHMARKS.keys()), index=0)

if start_date >= end_date:
    st.error("La fecha de inicio debe ser anterior a la final.")
    st.stop()

benchmark_series, benchmark_ticker, benchmark_source, benchmark_tried = resolve_series(
    BENCHMARKS[benchmark_label]["yahoo"],
    BENCHMARKS[benchmark_label]["stooq"],
    start_date,
    end_date,
)
if benchmark_series is None:
    st.error("No se pudo descargar el benchmark seleccionado.")
    st.code("\n".join(benchmark_tried))
    st.stop()

portfolio_labels = ["A", "B", "C"]
portfolio_colors = {"A": "#f0a128", "B": "#38d1f4", "C": "#ae8cff"}
portfolio_data = {}
cols = st.columns(3)

for col, label in zip(cols, portfolio_labels):
    with col:
        st.markdown(f"<div class='panel'><div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'><div class='port-title-{label.lower()}'>CARTERA {label}</div><div style='color:#2ce38a;font-weight:800'>100%</div></div>", unsafe_allow_html=True)
        count = st.number_input(f"NÂº fondos cartera {label}", min_value=1, max_value=5, value=1 if label == "A" else 3, step=1, key=f"n_{label}")
        funds = []
        weights = []
        for i in range(count):
            fund_name = st.selectbox(
                f"Fondo {i+1} ({label})",
                [f["nombre"] for f in ALL_FUNDS],
                index=min(i, len(ALL_FUNDS)-1),
                key=f"fund_{label}_{i}"
            )
            weight = st.number_input(f"Peso % ({label}-{i+1})", min_value=0.0, max_value=100.0, value=round(100.0/count, 1), step=0.1, key=f"w_{label}_{i}")
            funds.append(fund_name)
            weights.append(weight)
        total_w = sum(weights)
        st.caption(f"Total asignado: {total_w:.1f}%")
        portfolio_data[label] = {"funds": funds, "weights": weights, "total": total_w}
        st.markdown("</div>", unsafe_allow_html=True)

invalid = [k for k,v in portfolio_data.items() if abs(v["total"] - 100) > 0.2]
if invalid:
    st.warning(f"Las carteras {', '.join(invalid)} no suman 100%. Ajusta los pesos antes de interpretar resultados.")

loaded_series = {}
load_log = []

for p_label, p in portfolio_data.items():
    for fund_name in p["funds"]:
        if fund_name in loaded_series:
            continue
        meta = NAME_TO_META[fund_name]
        s, ticker, source, tried = resolve_series(meta["yahoo"], meta["stooq"], start_date, end_date)
        if s is not None:
            loaded_series[fund_name] = {"series": s, "ticker": ticker, "source": source, "meta": meta}
        else:
            load_log.append((fund_name, tried))

if not loaded_series:
    st.error("No se ha podido descargar ningÃºn fondo de los seleccionados.")
    st.stop()

series_map = {benchmark_label: benchmark_series}
for k, v in loaded_series.items():
    series_map[k] = v["series"]

prices_all = pd.concat(series_map, axis=1).dropna(how="all").sort_index().ffill().dropna()
if len(prices_all) < 30:
    st.error("No hay suficiente histÃ³rico comÃºn despuÃ©s de alinear fechas.")
    st.stop()

portfolio_series = {}
portfolio_cost = {}
portfolio_components = {}
for p_label, p in portfolio_data.items():
    available = []
    weights = []
    for fund_name, weight in zip(p["funds"], p["weights"]):
        if fund_name in prices_all.columns:
            available.append(fund_name)
            weights.append(weight)
    if not available:
        continue
    prices_subset = prices_all[available].dropna()
    weights_dict = dict(zip(available, weights))
    portfolio_series[p_label] = calc_portfolio_value(prices_subset, weights_dict)
    portfolio_cost[p_label] = sum(NAME_TO_META[x]["ter"] * w/100 for x, w in weights_dict.items())
    portfolio_components[p_label] = available

if not portfolio_series:
    st.error("No se pudo construir ninguna cartera con datos vÃ¡lidos.")
    st.stop()

port_df = pd.DataFrame(portfolio_series).dropna().sort_index()
common_benchmark = prices_all[benchmark_label].reindex(port_df.index).dropna()
port_df = port_df.loc[common_benchmark.index]

returns = port_df.pct_change().dropna()
bench_rets = common_benchmark.pct_change().dropna().reindex(returns.index)
port_gap = port_df.div(common_benchmark / common_benchmark.iloc[0] * 100, axis=0) - 1
port_dd = port_df.apply(calc_drawdown)
consistency = rolling_consistency(portfolio_series, window_months=36)

summary_rows = []
for p in port_df.columns:
    td_final = (port_df[p].iloc[-1] / ((common_benchmark.iloc[-1] / common_benchmark.iloc[0]) * 100) - 1) * 100
    active = returns[p] - bench_rets
    summary_rows.append({
        "Cartera": p,
        "Rent. total (%)": round((port_df[p].iloc[-1] / 100 - 1) * 100, 2),
        "Rent. anualizada (%)": round(annualized_return(port_df[p]), 2),
        "Tracking diff (%)": round(td_final, 2),
        "Tracking error (%)": round(active.std() * np.sqrt(252) * 100, 2),
        "MÃ¡x drawdown (%)": round(port_dd[p].min(), 2),
        "TER blend (%)": round(portfolio_cost.get(p, np.nan), 3),
        "Fondos vÃ¡lidos": len(portfolio_components.get(p, [])),
    })
summary = pd.DataFrame(summary_rows).set_index("Cartera")

best_return = summary["Rent. total (%)"].idxmax()
lowest_dd = summary["MÃ¡x drawdown (%)"].idxmax()
cheapest = summary["TER blend (%)"].idxmin()
closest = summary["Tracking diff (%)"].abs().idxmin()
cons_label = max(consistency, key=consistency.get) if consistency else None
cons_text = f"; la mÃ¡s consistente, <b>Cartera {cons_label}</b> ({consistency[cons_label]:.0f}% de los tramos)" if cons_label else ""

st.markdown(
    f"<div class='verdict'><b style='color:#ffbe4d'>Veredicto.</b> Por rentabilidad manda <b>Cartera {best_return}</b>; la que menos cae es <b>Cartera {lowest_dd}</b>; la mÃ¡s barata es <b>Cartera {cheapest}</b>; la mÃ¡s pegada al benchmark es <b>Cartera {closest}</b>{cons_text}.</div>",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Benchmark usado", benchmark_ticker)
m2.metric("Fuente benchmark", benchmark_source)
m3.metric("Observaciones", f"{len(port_df):,}".replace(",", "."))
m4.metric("Carteras vÃ¡lidas", str(len(port_df.columns)))

if load_log:
    with st.expander("Fondos no descargados"):
        for name, tried in load_log:
            st.write(f"**{name}**")
            st.code("\n".join(tried))

left, right = st.columns([1.1, 2.2])
with left:
    st.markdown("<div class='panel'><div class='smallcap'>Fondos detectados</div></div>", unsafe_allow_html=True)
    loaded_info = []
    for name, info in loaded_series.items():
        loaded_info.append({
            "Fondo": name,
            "ISIN": info["meta"]["isin"],
            "TER (%)": info["meta"]["ter"],
            "Ticker": info["ticker"],
            "Fuente": info["source"],
        })
    st.dataframe(pd.DataFrame(loaded_info), use_container_width=True)

with right:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_df.index, y=(common_benchmark/common_benchmark.iloc[0])*100, mode='lines', name='Benchmark', line=dict(color='#98aed2', width=2.4)))
    for p in port_df.columns:
        fig.add_trace(go.Scatter(x=port_df.index, y=port_df[p], mode='lines', name=f'Cartera {p}', line=dict(color=portfolio_colors[p], width=2.8 if p == best_return else 2.2)))
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,16,32,0.55)', height=400, margin=dict(l=10,r=10,t=20,b=10), legend=dict(orientation='h', y=1.08), yaxis_title='Base 100')
    fig.update_xaxes(gridcolor='rgba(120,155,220,.08)')
    fig.update_yaxes(gridcolor='rgba(120,155,220,.08)')
    st.subheader("Crecimiento de 100â‚¬")
    st.plotly_chart(fig, use_container_width=True)

if consistency:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Consistencia")
    st.caption("Ventanas mÃ³viles de 3 aÃ±os: porcentaje de tramos solapados en los que cada cartera fue la mejor.")
    for p in ['A','B','C']:
        if p in consistency:
            st.progress(min(max(consistency[p] / 100, 0), 1), text=f"Cartera {p}: {consistency[p]:.0f}%")
    st.markdown("</div>", unsafe_allow_html=True)

cA, cB = st.columns(2)
with cA:
    fig_gap = go.Figure()
    for p in port_gap.columns:
        fig_gap.add_trace(go.Scatter(x=port_gap.index, y=port_gap[p]*100, mode='lines', name=f'Cartera {p}', line=dict(color=portfolio_colors[p], width=2.2)))
    fig_gap.add_hline(y=0, line_dash='dash', line_color='rgba(220,230,255,.4)')
    fig_gap.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,16,32,0.55)', height=320, margin=dict(l=10,r=10,t=20,b=10), legend=dict(orientation='h', y=1.08), yaxis_title='Gap vs benchmark (%)')
    fig_gap.update_xaxes(gridcolor='rgba(120,155,220,.08)')
    fig_gap.update_yaxes(gridcolor='rgba(120,155,220,.08)')
    st.subheader("SeparaciÃ³n frente al benchmark")
    st.plotly_chart(fig_gap, use_container_width=True)
with cB:
    fig_dd = go.Figure()
    for p in port_dd.columns:
        fig_dd.add_trace(go.Scatter(x=port_dd.index, y=port_dd[p], mode='lines', name=f'Cartera {p}', line=dict(color=portfolio_colors[p], width=2.2), fill='tozeroy', fillcolor='rgba(180,190,255,.03)'))
    fig_dd.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,16,32,0.55)', height=320, margin=dict(l=10,r=10,t=20,b=10), legend=dict(orientation='h', y=1.08), yaxis_title='Drawdown (%)')
    fig_dd.update_xaxes(gridcolor='rgba(120,155,220,.08)')
    fig_dd.update_yaxes(gridcolor='rgba(120,155,220,.08)')
    st.subheader("CaÃ­das desde mÃ¡ximos")
    st.plotly_chart(fig_dd, use_container_width=True)

fig_bar = go.Figure()
for p in summary.index:
    fig_bar.add_trace(go.Bar(name=f"Cartera {p}", x=['Rent. total', 'Rent. anualizada', 'TER blend'], y=[summary.loc[p, 'Rent. total (%)'], summary.loc[p, 'Rent. anualizada (%)'], summary.loc[p, 'TER blend (%)']], marker_color=portfolio_colors[p]))
fig_bar.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,16,32,0.55)', height=320, margin=dict(l=10,r=10,t=20,b=10), barmode='group', legend=dict(orientation='h', y=1.08), yaxis_title='%')
fig_bar.update_xaxes(gridcolor='rgba(120,155,220,.08)')
fig_bar.update_yaxes(gridcolor='rgba(120,155,220,.08)')
st.subheader("Resumen comparado")
st.plotly_chart(fig_bar, use_container_width=True)

st.dataframe(summary, use_container_width=True)

st.caption(
    f"La interfaz replica la estructura visual de la web HTML de referencia descargada desde tu enlace, con tarjetas A/B/C, bloque de veredicto y paneles oscuros. "
    f"Los datos reales se descargan con yfinance y fallback a Stooq para benchmark/proxies; Yahoo Finance muestra sÃ­mbolos como URTH, IWDA.AS y SWDA.L, y Stooq publica histÃ³rico para R2.F e IWDA.UK. [web:26][web:27][web:33][web:61][web:62]"
)