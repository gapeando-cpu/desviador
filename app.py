import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import os
from io import StringIO
from datetime import date, datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Comparador de carteras — Fondos", layout="wide")

SAVE_FILE = "saved_portfolios.json"

BASE_CATALOG = [
    {
        "nombre": "iShares Developed World Index",
        "isin": "IE000ZYRH0Q7",
        "ter": 0.06,
        "yahoo": ["IE000ZYRH0Q7", "IE000ZYRH0Q7.IR", "IE000ZYRH0Q7.SG", "IE000ZYRH0Q7.F"],
        "stooq": [],
    },
    {
        "nombre": "Fidelity MSCI World Index Fund",
        "isin": "IE00BYX5NX33",
        "ter": 0.12,
        "yahoo": ["IE00BYX5NX33.SG", "IE00BYX5NX33"],
        "stooq": [],
    },
    {
        "nombre": "Vanguard Global Stock Index Fund",
        "isin": "IE00B03HD191",
        "ter": 0.18,
        "yahoo": ["IE00B03HD191.IR", "IE00B03HD191"],
        "stooq": [],
    },
    {
        "nombre": "MSCI World proxy ETF",
        "isin": "URTH",
        "ter": 0.24,
        "yahoo": ["URTH", "IWDA.AS", "SWDA.L"],
        "stooq": ["iwda.uk"],
    },
    {
        "nombre": "Amundi MSCI World UCITS ETF",
        "isin": "LU1681043599",
        "ter": 0.12,
        "yahoo": ["CW8.PA"],
        "stooq": [],
    },
]

PRESETS = {
    "A": [{"nombre": "iShares Developed World Index", "peso": 100.0}],
    "B": [
        {"nombre": "Fidelity MSCI World Index Fund", "peso": 50.0},
        {"nombre": "Vanguard Global Stock Index Fund", "peso": 50.0},
    ],
    "C": [
        {"nombre": "iShares Developed World Index", "peso": 40.0},
        {"nombre": "Fidelity MSCI World Index Fund", "peso": 20.0},
        {"nombre": "Vanguard Global Stock Index Fund", "peso": 40.0},
    ],
}

BENCHMARKS = {
    "MSCI World (índice Stooq)": {"yahoo": [], "stooq": ["r2.f"]},
    "MSCI World (proxy ETF Yahoo)": {"yahoo": ["URTH", "IWDA.AS", "SWDA.L"], "stooq": ["iwda.uk"]},
}

PORT_COLORS = {"A": "#f0a128", "B": "#38d1f4", "C": "#ae8cff"}


def load_saved_portfolios():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_saved_portfolios(data):
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_catalog():
    catalog = BASE_CATALOG.copy()
    customs = st.session_state.get("custom_catalog", [])
    existing = {x["nombre"] for x in catalog}
    for item in customs:
        if item["nombre"] not in existing:
            catalog.append(item)
    return catalog


def name_to_meta():
    return {f["nombre"]: f for f in get_catalog()}


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


def calc_drawdown(series):
    return (series / series.cummax() - 1) * 100


def tracking_error(active_returns):
    ar = active_returns.dropna()
    if ar.empty:
        return np.nan
    return ar.std() * np.sqrt(252) * 100


def calc_portfolio_value(prices, weights):
    w = pd.Series(weights, index=prices.columns, dtype=float)
    w = w / w.sum()
    normalized = prices / prices.iloc[0]
    return (normalized * w).sum(axis=1) * 100


def rolling_consistency(series_dict, window_months=36):
    monthly = pd.DataFrame(series_dict).resample("ME").last().dropna()
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


def autocomplete_options(query, catalog):
    if not query:
        return [f["nombre"] for f in catalog]
    q = query.lower().strip()
    ranked = []
    for f in catalog:
        hay = f"{f['nombre']} {f['isin']} {' '.join(f['yahoo'])}".lower()
        if q in hay:
            ranked.append(f["nombre"])
    return ranked if ranked else [f["nombre"] for f in catalog]


def normalize_weights(rows):
    total = sum(max(0, float(r["peso"])) for r in rows)
    if total <= 0:
        return rows
    for r in rows:
        r["peso"] = round(r["peso"] / total * 100, 2)
    return rows


if "saved_portfolios" not in st.session_state:
    st.session_state.saved_portfolios = load_saved_portfolios()
if "portfolio_rows" not in st.session_state:
    st.session_state.portfolio_rows = {k: [row.copy() for row in PRESETS[k]] for k in ["A", "B", "C"]}
if "search_terms" not in st.session_state:
    st.session_state.search_terms = {"A": "", "B": "", "C": ""}
if "custom_catalog" not in st.session_state:
    st.session_state.custom_catalog = []

st.markdown(
    """
<style>
.stApp {background: radial-gradient(circle at top right, rgba(58,93,171,.12), transparent 28%), linear-gradient(180deg, #05070d 0%, #070b14 35%, #050811 100%); color: #ecf2ff;}
.block-container {max-width: 1380px; padding-top: 1rem; padding-bottom: 2rem;}
h1,h2,h3 {letter-spacing: -.02em;}
.panel {background: linear-gradient(180deg, rgba(14,20,35,.98), rgba(9,14,26,.98)); border: 1px solid #1b2943; border-radius: 14px; padding: 12px 14px; box-shadow: 0 12px 32px rgba(0,0,0,.28);}
.verdict {border: 1px solid rgba(240,161,40,.55); border-radius: 11px; background: linear-gradient(180deg, rgba(34,24,7,.66), rgba(24,18,8,.72)); color: #f6f0de; padding: 12px 14px; font-size: 14px;}
.smallcap {font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:#9caecc; font-weight:700;}
.port-a {color:#f0a128; font-weight:800;}
.port-b {color:#38d1f4; font-weight:800;}
.port-c {color:#ae8cff; font-weight:800;}
div[data-testid="stMetric"] {background: linear-gradient(180deg,#101827,#0c1322); border:1px solid #1f2c45; border-radius: 14px; padding: 8px 10px;}
div[data-testid="stDataFrame"] {border:1px solid #1f2c45; border-radius:14px; overflow:hidden;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Comparador de carteras — Fondos")
st.caption(
    "Versión fina: elige fondos, añade tickers propios, guarda carteras, exporta CSV y compara con datos reales manteniendo el estilo del comparador original."
)

catalog = get_catalog()
meta_map = name_to_meta()
all_names = [f["nombre"] for f in catalog]

st.markdown(
    "<div class='panel'><div class='smallcap'>Empieza rápido</div><div style='font-size:12px;color:#90a3c8'>Configura tus carteras A, B y C con fondos reales, pesos personalizados y benchmark opcional. Puedes añadir tickers tuyos y guardar tus combinaciones favoritas.</div></div>",
    unsafe_allow_html=True,
)

with st.expander("Añadir fondo/ticker manual"):
    cc1, cc2, cc3, cc4 = st.columns([2, 1.2, 1, 1])
    custom_name = cc1.text_input("Nombre visible", placeholder="Ej. Amundi MSCI World")
    custom_ticker = cc2.text_input("Ticker Yahoo", placeholder="Ej. CW8.PA")
    custom_isin = cc3.text_input("ISIN opcional")
    custom_ter = cc4.number_input("TER (%)", min_value=0.0, max_value=5.0, value=0.12, step=0.01)
    if st.button("Añadir al catálogo"):
        if custom_name and custom_ticker:
            st.session_state.custom_catalog.append(
                {
                    "nombre": custom_name,
                    "isin": custom_isin if custom_isin else custom_ticker,
                    "ter": custom_ter,
                    "yahoo": [custom_ticker],
                    "stooq": [],
                }
            )
            st.success(f"Añadido: {custom_name}")
            st.rerun()
        else:
            st.warning("Pon al menos nombre y ticker.")

c0, c1, c2 = st.columns([1, 1, 1])
with c0:
    start_date = st.date_input("Fecha inicio", value=date(2021, 7, 1), key="start")
with c1:
    end_date = st.date_input("Fecha fin", value=datetime.today().date(), key="end")
with c2:
    benchmark_label = st.selectbox("Benchmark", list(BENCHMARKS.keys()), index=0)

saved_names = list(st.session_state.saved_portfolios.keys())
with st.expander("Carteras guardadas"):
    if saved_names:
        st.write(", ".join(saved_names))
    else:
        st.caption("Aún no hay ninguna. Guarda una cartera desde su panel.")

cols = st.columns(3)
for idx, label in enumerate(["A", "B", "C"]):
    with cols[idx]:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        total_pct = sum(r["peso"] for r in st.session_state.portfolio_rows[label])
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'><div class='port-{label.lower()}'>CARTERA {label}</div><div style='color:#2ce38a;font-weight:800'>{total_pct:.1f}%</div></div>",
            unsafe_allow_html=True,
        )
        portfolio_name = st.text_input(f"Nombre cartera {label}", value=f"Cartera {label}", key=f"portfolio_name_{label}")
        st.session_state.search_terms[label] = st.text_input(
            f"Buscar fondo para {label}",
            value=st.session_state.search_terms[label],
            key=f"search_{label}",
        )
        options = autocomplete_options(st.session_state.search_terms[label], catalog)
        add_col1, add_col2 = st.columns([3, 1])
        with add_col1:
            selected_to_add = st.selectbox(f"Selecciona fondo ({label})", options, key=f"add_select_{label}")
        with add_col2:
            if st.button(f"Añadir {label}"):
                st.session_state.portfolio_rows[label].append({"nombre": selected_to_add, "peso": 0.0})
                st.rerun()

        remove_idx = None
        for i, row in enumerate(st.session_state.portfolio_rows[label]):
            r1, r2, r3 = st.columns([3.3, 1, 0.45])
            safe_idx = all_names.index(row["nombre"]) if row["nombre"] in all_names else 0
            row["nombre"] = r1.selectbox(
                f"Fondo {label}-{i+1}",
                all_names,
                index=safe_idx,
                key=f"row_name_{label}_{i}",
            )
            row["peso"] = r2.number_input(
                f"Peso {label}-{i+1}",
                min_value=0.0,
                max_value=100.0,
                value=float(row["peso"]),
                step=0.1,
                key=f"row_weight_{label}_{i}",
            )
            if r3.button("×", key=f"del_{label}_{i}"):
                remove_idx = i

        if remove_idx is not None and len(st.session_state.portfolio_rows[label]) > 1:
            st.session_state.portfolio_rows[label].pop(remove_idx)
            st.rerun()

        ac1, ac2, ac3, ac4 = st.columns([1, 1, 1.3, 1])
        if ac1.button(f"Igualar {label}"):
            n = len(st.session_state.portfolio_rows[label])
            for row in st.session_state.portfolio_rows[label]:
                row["peso"] = round(100 / n, 2)
            st.rerun()

        if ac2.button(f"Vaciar {label}"):
            first_name = st.session_state.portfolio_rows[label][0]["nombre"]
            st.session_state.portfolio_rows[label] = [{"nombre": first_name, "peso": 100.0}]
            st.rerun()

        dup_target = {"A": "B", "B": "C", "C": "A"}[label]
        if ac3.button(f"Duplicar {label} → {dup_target}"):
            st.session_state.portfolio_rows[dup_target] = [r.copy() for r in st.session_state.portfolio_rows[label]]
            st.rerun()

        if ac4.button(f"Guardar {label}"):
            st.session_state.saved_portfolios[portfolio_name] = [r.copy() for r in st.session_state.portfolio_rows[label]]
            save_saved_portfolios(st.session_state.saved_portfolios)
            st.success(f"Guardada: {portfolio_name}")

        if saved_names:
            load_name = st.selectbox(f"Cargar guardada en {label}", [""] + saved_names, key=f"load_saved_{label}")
            if load_name and st.button(f"Aplicar en {label}"):
                st.session_state.portfolio_rows[label] = [r.copy() for r in st.session_state.saved_portfolios[load_name]]
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

b1, b2, b3, b4 = st.columns([1.4, 1.2, 1.2, 1.2])
with b1:
    selected_period = st.radio("Periodo", ["3A", "5A", "10A", "Máx"], horizontal=True, index=1)
with b2:
    hide_incomplete = st.checkbox("Ocultar fondos sin datos", value=True)
with b3:
    if st.button("Repartir al 100%"):
        for p in ["A", "B", "C"]:
            st.session_state.portfolio_rows[p] = normalize_weights(st.session_state.portfolio_rows[p])
        st.rerun()
with b4:
    if st.button("Limpiar todo"):
        st.session_state.portfolio_rows = {k: [row.copy() for row in PRESETS[k]] for k in ["A", "B", "C"]}
        st.rerun()

if start_date >= end_date:
    st.error("La fecha de inicio debe ser anterior a la final.")
    st.stop()

today = datetime.today().date()
if selected_period == "3A":
    start_date = max(start_date, date(today.year - 3, today.month, 1))
elif selected_period == "5A":
    start_date = max(start_date, date(today.year - 5, today.month, 1))
elif selected_period == "10A":
    start_date = max(start_date, date(today.year - 10, today.month, 1))

benchmark_series, benchmark_ticker, benchmark_source, benchmark_tried = resolve_series(
    BENCHMARKS[benchmark_label]["yahoo"],
    BENCHMARKS[benchmark_label]["stooq"],
    start_date,
    end_date,
)

if benchmark_series is None:
    st.error("No se pudo descargar el benchmark seleccionado.")
    st.code("
".join(benchmark_tried))
    st.stop()

loaded_series = {}
load_log = []
for label in ["A", "B", "C"]:
    for row in st.session_state.portfolio_rows[label]:
        fund_name = row["nombre"]
        if fund_name in loaded_series:
            continue
        meta = meta_map[fund_name]
        s, ticker, source, tried = resolve_series(meta["yahoo"], meta["stooq"], start_date, end_date)
        if s is not None:
            loaded_series[fund_name] = {
                "series": s,
                "ticker": ticker,
                "source": source,
                "meta": meta,
            }
        else:
            load_log.append((fund_name, tried))

if not loaded_series:
    st.error("No se ha podido descargar ningún fondo de los seleccionados.")
    st.stop()

series_map = {benchmark_label: benchmark_series}
for name, info in loaded_series.items():
    series_map[name] = info["series"]

prices_all = pd.concat(series_map, axis=1).dropna(how="all").sort_index().ffill().dropna()

if len(prices_all) < 30:
    st.error("No hay suficiente histórico común después de alinear fechas.")
    st.stop()

portfolio_series = {}
portfolio_cost = {}
portfolio_components = {}
portfolio_invalid = {}

for label in ["A", "B", "C"]:
    rows = st.session_state.portfolio_rows[label]
    valid = []
    weights = []
    invalid_names = []

    for row in rows:
        if row["nombre"] in prices_all.columns:
            valid.append(row["nombre"])
            weights.append(row["peso"])
        else:
            invalid_names.append(row["nombre"])

    portfolio_invalid[label] = invalid_names

    if valid and sum(weights) > 0:
        subset = prices_all[valid].dropna()
        portfolio_series[label] = calc_portfolio_value(subset, dict(zip(valid, weights)))
        portfolio_cost[label] = sum(meta_map[n]["ter"] * w / 100 for n, w in zip(valid, weights))
        portfolio_components[label] = valid

if not portfolio_series:
    st.error("No se pudo construir ninguna cartera con datos válidos.")
    st.stop()

port_df = pd.DataFrame(portfolio_series).dropna().sort_index()
bench_norm = prices_all[benchmark_label].reindex(port_df.index).dropna()
port_df = port_df.loc[bench_norm.index]
bench_base100 = bench_norm / bench_norm.iloc[0] * 100
returns = port_df.pct_change().dropna()
bench_rets = bench_norm.pct_change().dropna().reindex(returns.index)
port_gap = port_df.div(bench_base100, axis=0) - 1
port_dd = port_df.apply(calc_drawdown)
consistency = rolling_consistency(portfolio_series, window_months=36)

summary_rows = []
for p in port_df.columns:
    td_final = (port_df[p].iloc[-1] / bench_base100.iloc[-1] - 1) * 100
    active = returns[p] - bench_rets
    summary_rows.append(
        {
            "Cartera": p,
            "Rent. total (%)": round((port_df[p].iloc[-1] / 100 - 1) * 100, 2),
            "Rent. anualizada (%)": round(annualized_return(port_df[p]), 2),
            "Tracking diff (%)": round(td_final, 2),
            "Tracking error (%)": round(tracking_error(active), 2),
            "Máx drawdown (%)": round(port_dd[p].min(), 2),
            "TER blend (%)": round(portfolio_cost.get(p, np.nan), 3),
            "Fondos válidos": len(portfolio_components.get(p, [])),
        }
    )

summary = pd.DataFrame(summary_rows).set_index("Cartera")

best_return = summary["Rent. total (%)"].idxmax()
lowest_dd = summary["Máx drawdown (%)"].idxmax()
cheapest = summary["TER blend (%)"].idxmin()
closest = summary["Tracking diff (%)"].abs().idxmin()
cons_label = max(consistency, key=consistency.get) if consistency else None
cons_text = (
    f"; la más consistente, <b>Cartera {cons_label}</b> ({consistency[cons_label]:.0f}% de los tramos)"
    if cons_label
    else ""
)

st.markdown(
    f"<div class='verdict'><b style='color:#ffbe4d'>Veredicto.</b> "
    f"Por rentabilidad manda <b>Cartera {best_return}</b>; "
    f"la que menos cae es <b>Cartera {lowest_dd}</b>; "
    f"la más barata, <b>Cartera {cheapest}</b>; "
    f"la más pegada al benchmark, <b>Cartera {closest}</b>{cons_text}.</div>",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Benchmark usado", benchmark_ticker)
m2.metric("Fuente benchmark", benchmark_source)
m3.metric("Observaciones", f"{len(port_df):,}".replace(",", "."))
m4.metric("Carteras válidas", str(len(port_df.columns)))

if load_log and not hide_incomplete:
    with st.expander("Fondos no descargados"):
        for name, tried in load_log:
            st.write(f"**{name}**")
            st.code("
".join(tried))

for p in ["A", "B", "C"]:
    if portfolio_invalid.get(p):
        st.caption(f"Cartera {p}: sin datos para " + ", ".join(portfolio_invalid[p]))

left, right = st.columns([1.05, 2.2])

with left:
    loaded_info = []
    for name, info in loaded_series.items():
        loaded_info.append(
            {
                "Fondo": name,
                "ISIN": info["meta"]["isin"],
                "TER (%)": info["meta"]["ter"],
                "Ticker": info["ticker"],
                "Fuente": info["source"],
            }
        )
    st.dataframe(pd.DataFrame(loaded_info), use_container_width=True)

with right:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=port_df.index,
            y=bench_base100,
            mode="lines",
            name="Benchmark",
            line=dict(color="#98aed2", width=2.2),
        )
    )
    for p in port_df.columns:
        fig.add_trace(
            go.Scatter(
                x=port_df.index,
                y=port_df[p],
                mode="lines",
                name=f"Cartera {p}",
                line=dict(color=PORT_COLORS[p], width=2.9 if p == best_return else 2.2),
                fill="tozeroy" if p == best_return else None,
                fillcolor="rgba(255,255,255,0.02)",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,16,32,0.55)",
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.08),
        yaxis_title="Base 100",
    )
    fig.update_xaxes(gridcolor="rgba(120,155,220,.08)")
    fig.update_yaxes(gridcolor="rgba(120,155,220,.08)")
    st.subheader("Crecimiento de 100€")
    st.plotly_chart(fig, use_container_width=True)

if consistency:
    st.subheader("Consistencia")
    st.caption("Ventanas móviles de 3 años: porcentaje de tramos solapados en los que cada cartera fue la mejor.")
    for p in ["A", "B", "C"]:
        if p in consistency:
            st.progress(min(max(consistency[p] / 100, 0), 1), text=f"Cartera {p}: {consistency[p]:.0f}%")

cA, cB = st.columns(2)

with cA:
    fig_gap = go.Figure()
    for p in port_gap.columns:
        fig_gap.add_trace(
            go.Scatter(
                x=port_gap.index,
                y=port_gap[p] * 100,
                mode="lines",
                name=f"Cartera {p}",
                line=dict(color=PORT_COLORS[p], width=2.2),
            )
        )
    fig_gap.add_hline(y=0, line_dash="dash", line_color="rgba(220,230,255,.4)")
    fig_gap.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,16,32,0.55)",
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.08),
        yaxis_title="Gap vs benchmark (%)",
    )
    fig_gap.update_xaxes(gridcolor="rgba(120,155,220,.08)")
    fig_gap.update_yaxes(gridcolor="rgba(120,155,220,.08)")
    st.subheader("Separación frente al benchmark")
    st.plotly_chart(fig_gap, use_container_width=True)

with cB:
    fig_dd = go.Figure()
    for p in port_dd.columns:
        fig_dd.add_trace(
            go.Scatter(
                x=port_dd.index,
                y=port_dd[p],
                mode="lines",
                name=f"Cartera {p}",
                line=dict(color=PORT_COLORS[p], width=2.2),
                fill="tozeroy",
                fillcolor="rgba(180,190,255,.03)",
            )
        )
    fig_dd.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,16,32,0.55)",
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.08),
        yaxis_title="Drawdown (%)",
    )
    fig_dd.update_xaxes(gridcolor="rgba(120,155,220,.08)")
    fig_dd.update_yaxes(gridcolor="rgba(120,155,220,.08)")
    st.subheader("Caídas desde máximos")
    st.plotly_chart(fig_dd, use_container_width=True)

fig_bar = go.Figure()
for p in summary.index:
    fig_bar.add_trace(
        go.Bar(
            name=f"Cartera {p}",
            x=["Rent. total", "Rent. anualizada", "TER blend"],
            y=[
                summary.loc[p, "Rent. total (%)"],
                summary.loc[p, "Rent. anualizada (%)"],
                summary.loc[p, "TER blend (%)"],
            ],
            marker_color=PORT_COLORS[p],
        )
    )

fig_bar.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,16,32,0.55)",
    height=320,
    margin=dict(l=10, r=10, t=20, b=10),
    barmode="group",
    legend=dict(orientation="h", y=1.08),
    yaxis_title="%",
)
fig_bar.update_xaxes(gridcolor="rgba(120,155,220,.08)")
fig_bar.update_yaxes(gridcolor="rgba(120,155,220,.08)")
st.subheader("Resumen comparado")
st.plotly_chart(fig_bar, use_container_width=True)

st.dataframe(summary, use_container_width=True)

csv_summary = summary.reset_index().to_csv(index=False).encode("utf-8")
st.download_button(
    "Descargar resumen CSV",
    csv_summary,
    file_name="comparador_carteras_resumen.csv",
    mime="text/csv",
)