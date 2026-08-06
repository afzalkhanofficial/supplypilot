"""
SupplyPilot Streamlit Dashboard.

An interactive dark-mode dashboard providing multi-page access to supply chain
operations:

Page 1 — Overview: High-level KPI metrics, fleet-wide risk distribution, and quick table.
Page 2 — Inventory: Detailed stock status, safety stock, EOQ, and reorder points per product.
Page 3 — Demand Forecast: Interactive Prophet demand projections with 80% CI bands.
Page 4 — Purchase Orders: Human-in-the-loop purchase order review and creation.
Page 5 — Agent Chat: Interactive assistant with multi-turn context and tool visibility.
Page 6 — Supplier Intelligence: RAG search and document ingestion for SLAs, contracts, policies.

Run locally:
    streamlit run dashboard/app.py --server.port 8501
"""

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path so modules resolve cleanly when running via `streamlit run`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.api_client import (
    APIError,
    agent_chat,
    approve_order,
    create_order,
    get_forecast,
    get_inventory,
    health,
    ingest_document,
    list_documents as list_docs,
    list_orders,
    list_products,
    reject_order,
    scan_inventory,
    search_documents as search_docs,
)


# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SupplyPilot — Supply Chain AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — Premium Glassmorphism & Enterprise Theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

/* ── Base App Theme ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
}
.stApp {
    background: radial-gradient(circle at 50% 0%, #0f172a 0%, #090d16 60%, #05070e 100%);
    color: #f1f5f9;
}

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070b14 0%, #0d1322 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
}

/* Hide raw radio circles */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input[type="radio"] {
    display: none !important;
}

/* Style radio labels as sleek navigation pills */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    padding: 11px 16px !important;
    margin-bottom: 6px !important;
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    color: #94a3b8 !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(56, 189, 248, 0.08) !important;
    color: #38bdf8 !important;
    border-color: rgba(56, 189, 248, 0.2) !important;
    transform: translateX(3px) !important;
}

/* Selected active radio tab */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(56, 189, 248, 0.16) 0%, rgba(129, 140, 248, 0.08) 100%) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 15px rgba(56, 189, 248, 0.15) !important;
}

/* ── Modern KPI Cards ── */
.kpi-card {
    background: linear-gradient(145deg, rgba(26, 36, 56, 0.65), rgba(15, 23, 42, 0.85));
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 22px 20px;
    text-align: left;
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.08);
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 40px -10px rgba(56,189,248,0.2), inset 0 1px 0 rgba(255,255,255,0.15);
    border-color: rgba(255, 255, 255, 0.15);
}
.kpi-card-blue   { border-top: 3px solid #38bdf8; }
.kpi-card-amber  { border-top: 3px solid #fbbf24; }
.kpi-card-red    { border-top: 3px solid #f43f5e; }
.kpi-card-green  { border-top: 3px solid #34d399; }
.kpi-card-purple { border-top: 3px solid #a855f7; }

.kpi-label { color: #94a3b8; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.5rem; font-weight: 800; line-height: 1.1; margin-bottom: 4px; letter-spacing: -0.03em; }
.kpi-sub   { color: #64748b; font-size: 0.75rem; font-weight: 500; }

.kpi-blue   { color: #38bdf8; }
.kpi-amber  { color: #fbbf24; }
.kpi-red    { color: #f43f5e; }
.kpi-green  { color: #34d399; }
.kpi-purple { color: #c084fc; }

/* ── Section Header ── */
.section-header {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.15rem; font-weight: 700; color: #f8fafc;
    display: flex; align-items: center; gap: 10px;
    margin: 28px 0 16px 0;
}
.section-header::before {
    content: '';
    display: inline-block;
    width: 4px; height: 18px;
    background: linear-gradient(180deg, #38bdf8, #818cf8);
    border-radius: 4px;
}

/* ── Page Titles ── */
.page-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.1rem; font-weight: 800; color: #f8fafc;
    letter-spacing: -0.03em; margin-bottom: 4px;
}
.page-sub { color: #94a3b8; font-size: 0.95rem; font-weight: 400; margin-bottom: 24px; }

/* ── Status Badges ── */
.badge { display: inline-flex; align-items: center; gap: 5px; padding: 4px 12px; border-radius: 999px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; }
.badge-critical { background: rgba(244, 63, 94, 0.12); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.35); box-shadow: 0 0 12px rgba(244, 63, 94, 0.15); }
.badge-warning  { background: rgba(251, 191, 36, 0.12); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.35); box-shadow: 0 0 12px rgba(251, 191, 36, 0.15); }
.badge-ok       { background: rgba(52, 211, 153, 0.12); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.35); box-shadow: 0 0 12px rgba(52, 211, 153, 0.15); }

/* ── HTML Styled Table ── */
.styled-table {
    width: 100%; border-collapse: separate; border-spacing: 0;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px; overflow: hidden;
    margin-top: 8px; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.4);
}
.styled-table th {
    background: rgba(30, 41, 59, 0.8);
    color: #94a3b8; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    padding: 14px 18px; text-align: left;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.styled-table td {
    padding: 14px 18px; color: #e2e8f0; font-size: 0.88rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    transition: background 0.15s;
}
.styled-table tr:last-child td { border-bottom: none; }
.styled-table tr:hover td { background: rgba(56, 189, 248, 0.05); }

/* ── Glassmorphism Chat Bubbles ── */
.chat-user {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(129, 140, 248, 0.1));
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 14px 14px 2px 14px;
    padding: 14px 18px; margin: 10px 0; color: #f8fafc;
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.1);
}
.chat-agent {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 14px 14px 14px 2px;
    padding: 14px 18px; margin: 10px 0; color: #cbd5e1;
    backdrop-filter: blur(12px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}
.tool-pill {
    display: inline-block; margin: 2px 4px;
    background: rgba(56, 189, 248, 0.12); color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 999px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600;
}
.agent-meta { color: #64748b; font-size: 0.75rem; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 6px; }

/* ── Streamlit Input Overrides ── */
.stTextInput input, .stSelectbox select, .stNumberInput input, .stTextArea textarea {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #f8fafc !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
}
.stTextInput input:focus, .stSelectbox select:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.25) !important;
}

/* ── Global Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.88rem !important; padding: 10px 22px !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.45) !important;
}

/* ── Tabs Styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: rgba(15, 23, 42, 0.5);
    padding: 6px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.06);
}
.stTabs [data-baseweb="tab"] {
    height: auto; padding: 10px 18px; border-radius: 8px;
    color: #94a3b8; font-weight: 600; font-size: 0.88rem;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(255, 255, 255, 0.08); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly dark layout defaults
# ---------------------------------------------------------------------------

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.4)",
    font=dict(family="Plus Jakarta Sans, Inter", color="#94a3b8"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _badge(risk: str) -> str:
    cls = {"CRITICAL": "badge-critical", "WARNING": "badge-warning", "OK": "badge-ok"}.get(risk, "badge-ok")
    return f'<span class="badge {cls}">{risk}</span>'


def _kpi(label: str, value: str, sub: str = "", color_class: str = "kpi-blue") -> str:
    card_type = {
        "kpi-blue": "kpi-card-blue",
        "kpi-amber": "kpi-card-amber",
        "kpi-red": "kpi-card-red",
        "kpi-green": "kpi-card-green",
        "kpi-purple": "kpi-card-purple",
    }.get(color_class, "kpi-card-blue")
    return f"""
    <div class="kpi-card {card_type}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {color_class}">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def _api_error(exc: APIError) -> None:
    st.error(f"**API Error:** {exc}")
    if "port 8000" in str(exc):
        st.info("Start the API server with: `.venv\\Scripts\\python scripts/run_api.py`")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 6px 20px 6px; text-align: left; border-bottom: 1px solid rgba(255, 255, 255, 0.07); margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: linear-gradient(135deg, #38bdf8, #818cf8); border-radius: 12px; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-size: 1.35rem; box-shadow: 0 8px 20px rgba(56, 189, 248, 0.3);">🚀</div>
            <div>
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 1.25rem; background: linear-gradient(135deg, #ffffff, #cbd5e1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.02em;">SupplyPilot</div>
                <div style="font-size: 0.68rem; font-weight: 700; color: #38bdf8; letter-spacing: 0.1em; text-transform: uppercase;">SUPPLY CHAIN AI</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "📊  Overview",
            "📦  Inventory",
            "📈  Demand Forecast",
            "🛒  Purchase Orders",
            "🤖  Agent Chat",
            "📄  Supplier Intelligence",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='margin: 20px 0 16px 0;'/>", unsafe_allow_html=True)

    # Live health indicator pill
    try:
        h = health()
        db_ok = h.get("db_connected", False)
        dot_color = "#34d399" if db_ok else "#f43f5e"
        st.markdown(
            f'''<div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 10px 14px; font-size: 0.78rem; color: #94a3b8;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>API Server</span>
                    <span style="color: {dot_color}; font-weight: 600;">● Online</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                    <span>PostgreSQL DB</span>
                    <span style="color: {dot_color}; font-weight: 600;">● Connected</span>
                </div>
            </div>''',
            unsafe_allow_html=True,
        )
    except APIError:
        st.markdown(
            '''<div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.25); border-radius: 10px; padding: 10px 14px; font-size: 0.78rem; color: #f43f5e; font-weight: 600;">
                ● API Backend Offline
            </div>''',
            unsafe_allow_html=True,
        )


# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================

if page == "📊  Overview":
    st.markdown('<div class="page-title">Fleet Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Real-time inventory health and risk distribution across all products</div>', unsafe_allow_html=True)

    try:
        scan = scan_inventory()
        counts = scan["counts"]
        summary = scan["summary"]

        orders_data = list_orders(status="pending", limit=200)
        pending_count = orders_data["total"]

        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(_kpi("Total Products", str(scan["scanned"]), "tracked in system", "kpi-blue"), unsafe_allow_html=True)
        c2.markdown(_kpi("Critical", str(counts["CRITICAL"]), "immediate action needed", "kpi-red"), unsafe_allow_html=True)
        c3.markdown(_kpi("Warning", str(counts["WARNING"]), "below reorder point", "kpi-amber"), unsafe_allow_html=True)
        c4.markdown(_kpi("Pending Orders", str(pending_count), "awaiting approval", "kpi-green"), unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Risk distribution bar chart
        st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
        fig_dist = go.Figure(go.Bar(
            x=["CRITICAL", "WARNING", "OK"],
            y=[counts["CRITICAL"], counts["WARNING"], counts["OK"]],
            marker=dict(
                color=["#f43f5e", "#fbbf24", "#34d399"],
                line=dict(color="rgba(255,255,255,0.15)", width=1),
            ),
            text=[counts["CRITICAL"], counts["WARNING"], counts["OK"]],
            textposition="auto",
        ))
        fig_dist.update_layout(**_PLOTLY_LAYOUT, title="Products by Risk Level", height=280)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Inventory summary table
        st.markdown('<div class="section-header">Product Risk Table</div>', unsafe_allow_html=True)

        df = pd.DataFrame(summary)
        if not df.empty:
            df["risk_badge"] = df["risk_level"].apply(_badge)
            df["days_of_cover"] = df["days_of_cover"].apply(lambda x: f"{x:.1f}d")
            df["current_stock"] = df["current_stock"].apply(lambda x: f"{x:,.0f}")
            df["reorder_point"] = df["reorder_point"].apply(lambda x: f"{x:,}")
            df["eoq"] = df["eoq"].apply(lambda x: f"{x:,}")

            display_df = df[["product_id", "risk_badge", "current_stock", "reorder_point", "eoq", "days_of_cover"]].rename(columns={
                "product_id": "Product ID",
                "risk_badge": "Risk Status",
                "current_stock": "Current Stock",
                "reorder_point": "Reorder Point",
                "eoq": "EOQ Qty",
                "days_of_cover": "Stock Cover",
            })
            st.write(display_df.to_html(escape=False, index=False, classes="styled-table"), unsafe_allow_html=True)

    except APIError as exc:
        _api_error(exc)


# ===========================================================================
# PAGE 2 — INVENTORY
# ===========================================================================

elif page == "📦  Inventory":
    st.markdown('<div class="page-title">Inventory Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Per-product stock levels, safety stock buffers, reorder points, and EOQ</div>', unsafe_allow_html=True)

    try:
        products = list_products()
        product_map = {f"Product {p['product_id']} — {p['product_name']}": p["product_id"] for p in products}
        selected_label = st.selectbox("Select product", list(product_map.keys()))
        pid = product_map[selected_label]

        inv = get_inventory(pid)

        risk = inv["risk_level"]
        color = {"CRITICAL": "kpi-red", "WARNING": "kpi-amber", "OK": "kpi-green"}[risk]

        st.markdown("<br/>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(_kpi("Current Stock", f"{inv['current_stock']:,.0f}", "units on hand", color), unsafe_allow_html=True)
        c2.markdown(_kpi("Days of Cover", f"{inv['days_of_cover']:.1f}d", f"risk: {risk}", color), unsafe_allow_html=True)
        c3.markdown(_kpi("Reorder Point", f"{inv['reorder_point']:,}", "units (trigger)", "kpi-amber"), unsafe_allow_html=True)
        c4.markdown(_kpi("EOQ", f"{inv['eoq']:,}", "optimal order qty", "kpi-blue"), unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        c5.markdown(_kpi("Safety Stock", f"{inv['safety_stock']:,}", "buffer (95% SL)", "kpi-blue"), unsafe_allow_html=True)
        c6.markdown(_kpi("Lead Time", f"{inv['lead_time_days']}d", "supplier lead time", "kpi-blue"), unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Recommendation callout box
        rec_color = {"CRITICAL": "#f43f5e", "WARNING": "#fbbf24", "OK": "#34d399"}[risk]
        st.markdown(f"""
        <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(255,255,255,0.08); border-left: 4px solid {rec_color}; border-radius: 14px; padding: 20px; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);">
            <div style="font-weight: 700; color: {rec_color}; font-size: 1.05rem; margin-bottom: 6px;">Recommendation: {inv['action']}</div>
            <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.5;">
                Current stock is <b>{inv['current_stock']:,} units</b> vs Reorder Point <b>{inv['reorder_point']:,} units</b>. 
                Recommended order size (EOQ): <b>{inv['eoq']:,} units</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    except APIError as exc:
        _api_error(exc)


# ===========================================================================
# PAGE 3 — DEMAND FORECAST
# ===========================================================================

elif page == "📈  Demand Forecast":
    st.markdown('<div class="page-title">Demand Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Facebook Prophet time-series predictions with 80% confidence intervals</div>', unsafe_allow_html=True)

    try:
        products = list_products()
        product_map = {f"Product {p['product_id']} — {p['product_name']}": p["product_id"] for p in products}

        col_sel, col_days = st.columns([3, 2])
        with col_sel:
            selected_label = st.selectbox("Select product", list(product_map.keys()))
            pid = product_map[selected_label]
        with col_days:
            days_ahead = st.slider("Days ahead", min_value=7, max_value=90, value=30)

        fc = get_forecast(pid, days_ahead=days_ahead)

        dates = fc["dates"]
        yhat = fc["yhat"]
        yhat_lower = fc["yhat_lower"]
        yhat_upper = fc["yhat_upper"]

        total_demand = fc["total_forecast"]
        daily_avg = total_demand / max(len(dates), 1)
        cutoff = fc["training_end"]

        st.markdown("<br/>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(_kpi("Total Forecast", f"{total_demand:,.0f}", f"over next {days_ahead} days", "kpi-blue"), unsafe_allow_html=True)
        c2.markdown(_kpi("Daily Average", f"{daily_avg:,.0f}", "units/day", "kpi-blue"), unsafe_allow_html=True)
        c3.markdown(_kpi("Model Cut-off", cutoff, "last historical date", "kpi-purple"), unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Plotly chart
        st.markdown(f'<div class="section-header">Forecast Chart — Product {pid}</div>', unsafe_allow_html=True)


        fig = go.Figure()

        # 80% Confidence Interval Band
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=yhat_upper + yhat_lower[::-1],
            fill="todense",
            fillcolor="rgba(56, 189, 248, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="80% CI Band",
        ))

        # Main Forecast Line
        fig.add_trace(go.Scatter(
            x=dates,
            y=yhat,
            mode="lines",
            name="Forecast (yhat)",
            line=dict(color="#38bdf8", width=3),
            hovertemplate="<b>%{x}</b><br>Forecast: %{y:,.0f} units<extra></extra>",
        ))

        fig.update_layout(**_PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig, use_container_width=True)

    except APIError as exc:
        _api_error(exc)


# ===========================================================================
# PAGE 4 — PURCHASE ORDERS
# ===========================================================================

elif page == "🛒  Purchase Orders":
    st.markdown('<div class="page-title">Purchase Orders</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Review pending orders and create new replenishment requests</div>', unsafe_allow_html=True)

    tab_pending, tab_all, tab_create = st.tabs([
        "⏳  Pending Orders",
        "📋  All Orders",
        "➕  Create Order",
    ])

    with tab_pending:
        try:
            orders_res = list_orders(status="pending", limit=100)
            pending_orders = orders_res.get("orders", [])

            if not pending_orders:
                st.info("No pending purchase orders awaiting approval.")
            else:
                st.markdown(f"**{len(pending_orders)} pending order(s) require review:**")
                for po in pending_orders:
                    st.markdown(f"""
                    <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(251,191,36,0.3); border-radius: 14px; padding: 18px; margin-bottom: 12px; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.4);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-weight: 700; color: #f8fafc; font-size: 1.05rem;">
                                Order #{po['id']} — Product {po['product_id']}
                            </span>
                            <span class="badge badge-warning">PENDING</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;">
                            <b>Quantity:</b> {po['quantity']:,} units &nbsp;|&nbsp; 
                            <b>Total Cost:</b> ${po['total_cost']:,.2f} &nbsp;|&nbsp; 
                            <b>Supplier:</b> {po['supplier_name']}
                        </div>
                        <div style="font-size: 0.82rem; color: #cbd5e1; font-style: italic;">"{po['reason']}"</div>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_c1, btn_c2, _ = st.columns([1, 1, 4])
                    with btn_c1:
                        if st.button("✅ Approve", key=f"app_{po['id']}"):
                            try:
                                res = approve_order(po["id"])
                                st.success(res["message"])
                                st.rerun()
                            except APIError as exc:
                                _api_error(exc)
                    with btn_c2:
                        if st.button("❌ Reject", key=f"rej_{po['id']}"):
                            try:
                                res = reject_order(po["id"])
                                st.info(res["message"])
                                st.rerun()
                            except APIError as exc:
                                _api_error(exc)
        except APIError as exc:
            _api_error(exc)

    with tab_all:
        try:
            status_filter = st.selectbox("Filter by status", ["All", "pending", "approved", "rejected"])
            filter_param = None if status_filter == "All" else status_filter

            all_orders_res = list_orders(status=filter_param, limit=200)
            all_orders = all_orders_res.get("orders", [])

            if not all_orders:
                st.info("No orders found matching criteria.")
            else:
                df_orders = pd.DataFrame(all_orders)
                df_orders["status_badge"] = df_orders["status"].apply(lambda s: {
                    "pending": '<span class="badge badge-warning">PENDING</span>',
                    "approved": '<span class="badge badge-ok">APPROVED</span>',
                    "rejected": '<span class="badge badge-critical">REJECTED</span>',
                }.get(s, s))
                df_orders["total_cost"] = df_orders["total_cost"].apply(lambda c: f"${c:,.2f}")
                df_orders["quantity"] = df_orders["quantity"].apply(lambda q: f"{q:,}")

                display_df = df_orders[["id", "product_id", "quantity", "supplier_name", "total_cost", "status_badge", "created_at"]].rename(columns={
                    "id": "Order ID",
                    "product_id": "Product",
                    "quantity": "Quantity",
                    "supplier_name": "Supplier",
                    "total_cost": "Total Cost",
                    "status_badge": "Status",
                    "created_at": "Created At",
                })
                st.write(display_df.to_html(escape=False, index=False, classes="styled-table"), unsafe_allow_html=True)
        except APIError as exc:
            _api_error(exc)

    with tab_create:
        try:
            products = list_products()
            product_map = {f"Product {p['product_id']} — {p['product_name']}": p["product_id"] for p in products}

            with st.form("create_po_form"):
                st.markdown('<div class="section-header">Create Purchase Order</div>', unsafe_allow_html=True)
                sel_prod = st.selectbox("Product", list(product_map.keys()))
                quantity = st.number_input("Quantity", min_value=1, value=500, step=50)
                reason = st.text_area("Reason", value="Stock replenishment order.")

                submitted = st.form_submit_button("➕ Create Purchase Order")
                if submitted:
                    pid = product_map[sel_prod]
                    try:
                        result = create_order(pid, int(quantity), reason.strip())
                        st.success(result["message"])
                    except APIError as exc:
                        _api_error(exc)
        except APIError as exc:
            _api_error(exc)


# ===========================================================================
# PAGE 5 — AGENT CHAT
# ===========================================================================

elif page == "🤖  Agent Chat":
    st.markdown('<div class="page-title">Agent Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Ask SupplyPilot anything about your supply chain</div>', unsafe_allow_html=True)

    # Initialise session state for conversation history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    prompt_to_process = None

    # Suggested prompts (shown when conversation is empty)
    if not st.session_state.chat_messages:
        st.markdown('<div class="section-header">Suggested questions</div>', unsafe_allow_html=True)
        suggestions = [
            "Which products are at WARNING or CRITICAL stock risk?",
            "Show me the inventory status for product 85.",
            "What is the 14-day demand forecast for product 262?",
            "What is the fill rate penalty for Apex Supply Co?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"sug_{i}"):
                prompt_to_process = s

    # Render existing conversation messages
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            meta = msg.get("meta", {})
            tools_html = " ".join(f'<span class="tool-pill">{t}</span>' for t in meta.get("tools_used", []))
            meta_line = ""
            if tools_html:
                meta_line = f'<div class="agent-meta">Tools: {tools_html} &nbsp;·&nbsp; Steps: {meta.get("steps", "?")}</div>'
            st.markdown(
                f'<div class="chat-agent">🤖 {msg["content"]}{meta_line}</div>',
                unsafe_allow_html=True,
            )

    # Native chat input box
    user_typed_input = st.chat_input("Ask SupplyPilot a question...")
    if user_typed_input and user_typed_input.strip():
        prompt_to_process = user_typed_input.strip()

    # Execute agent call if user typed input or clicked a suggested question
    if prompt_to_process:
        st.session_state.chat_messages.append({"role": "user", "content": prompt_to_process})
        st.markdown(f'<div class="chat-user">🧑 {prompt_to_process}</div>', unsafe_allow_html=True)

        # Build chat_history for multi-turn context
        history_pairs = []
        msgs = st.session_state.chat_messages[:-1]
        i = 0
        while i < len(msgs) - 1:
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                history_pairs.append({"human": msgs[i]["content"], "ai": msgs[i + 1]["content"]})
                i += 2
            else:
                i += 1

        with st.spinner("SupplyPilot is thinking..."):
            try:
                result = agent_chat(prompt_to_process, chat_history=history_pairs)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "meta": {"tools_used": result.get("tools_used", []), "steps": result.get("steps", 0)},
                })
            except APIError as exc:
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": f"Error: {exc}",
                    "meta": {},
                })

        st.rerun()

    # Clear chat button
    if st.session_state.chat_messages:
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("🗑️  Clear conversation"):
            st.session_state.chat_messages = []
            st.rerun()


# ===========================================================================
# PAGE 6 — SUPPLIER INTELLIGENCE (RAG)
# ===========================================================================

elif page == "📄  Supplier Intelligence":
    st.markdown('<div class="page-title">Supplier Document Intelligence (RAG)</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Semantic search across contracts, SLAs, and policies powered by vector embeddings</div>', unsafe_allow_html=True)

    tab_search, tab_upload, tab_library = st.tabs([
        "🔍  Semantic Document Search",
        "📤  Upload & Ingest Document",
        "📁  Document Library",
    ])

    with tab_search:
        st.markdown('<div class="section-header">Ask or Search Supplier Documents</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns([3, 1.5, 1.5])
        with c1:
            q_input = st.text_input(
                "Search query",
                placeholder="e.g. What is the fill rate penalty for Apex Supply Co?",
                label_visibility="collapsed",
            )
        with c2:
            sup_filter = st.text_input(
                "Supplier filter (optional)",
                placeholder="All suppliers",
                label_visibility="collapsed",
            )
        with c3:
            doc_type_filter = st.selectbox(
                "Doc Type",
                ["All Types", "contract", "sla", "policy"],
                label_visibility="collapsed",
            )

        top_k_val = st.slider("Top results (K)", min_value=1, max_value=10, value=5)

        if st.button("🔎  Search Documents") or q_input:
            if not q_input.strip():
                st.warning("Please enter a search query.")
            else:
                with st.spinner("Embedding query & searching vector database..."):
                    try:
                        dtype_param = None if doc_type_filter == "All Types" else doc_type_filter
                        sup_param = sup_filter.strip() if sup_filter.strip() else None
                        search_res = search_docs(
                            query=q_input.strip(),
                            top_k=top_k_val,
                            supplier_name=sup_param,
                            doc_type=dtype_param,
                        )

                        if search_res.get("status") == "no_results":
                            st.info(search_res.get("message", "No matching document chunks found."))
                        elif search_res.get("status") == "ok":
                            results = search_res.get("results", [])
                            st.success(f"Found {len(results)} relevant document passage(s):")

                            for res in results:
                                sim_pct = int(res["similarity"] * 100)
                                st.markdown(f"""
                                <div style="background: rgba(15,23,42,0.85); border: 1px solid rgba(56,189,248,0.25); border-radius: 14px; padding: 18px; margin-bottom: 14px; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                        <span style="font-weight: 700; color: #38bdf8; font-size: 1.05rem;">
                                            📄 {res['filename']} (Rank #{res['rank']})
                                        </span>
                                        <span style="background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.3); border-radius: 999px; padding: 3px 12px; font-size: 0.78rem; font-weight: 700;">
                                            {sim_pct}% Relevance
                                        </span>
                                    </div>
                                    <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 12px;">
                                        <b>Supplier:</b> {res['supplier_name']} &nbsp;|&nbsp; <b>Type:</b> {res['doc_type'].upper()} &nbsp;|&nbsp; <b>Chunk:</b> #{res['chunk_index']}
                                    </div>
                                    <div style="background: rgba(9,13,22,0.9); border-left: 4px solid #38bdf8; padding: 14px 18px; border-radius: 8px; font-size: 0.9rem; line-height: 1.6; color: #e2e8f0; white-space: pre-wrap;">{res['chunk_text']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    except APIError as exc:
                        _api_error(exc)

    with tab_upload:
        st.markdown('<div class="section-header">Upload New Supplier Contract or SLA</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a document (.pdf or .txt)",
            type=["pdf", "txt"],
            help="PDFs will be extracted using PyPDF; text files will be read directly.",
        )

        c_up1, c_up2 = st.columns(2)
        with c_up1:
            up_supplier = st.text_input("Supplier Name", placeholder="e.g. Apex Supply Co")
        with c_up2:
            up_doc_type = st.selectbox("Document Type", ["contract", "sla", "policy"])

        if st.button("📤  Ingest & Vectorize Document"):
            if not uploaded_file:
                st.warning("Please select a file to upload.")
            elif not up_supplier.strip():
                st.warning("Please enter a supplier name.")
            else:
                with st.spinner("Extracting text, generating 384-dim vector embeddings & storing in pgvector..."):
                    try:
                        bytes_data = uploaded_file.read()
                        ingest_res = ingest_document(
                            file_name=uploaded_file.name,
                            file_bytes=bytes_data,
                            supplier_name=up_supplier.strip(),
                            doc_type=up_doc_type,
                        )

                        status = ingest_res.get("status")
                        if status == "ok":
                            st.success(
                                f"Successfully ingested **{ingest_res.get('filename')}**! "
                                f"Stored {ingest_res.get('chunks_stored')} text chunks "
                                f"(Doc ID: {ingest_res.get('document_id')})."
                            )
                        elif status == "duplicate":
                            st.info(f"ℹ️ {ingest_res.get('message')}")
                        else:
                            st.error(f"Error: {ingest_res.get('message')}")
                    except APIError as exc:
                        _api_error(exc)

    with tab_library:
        st.markdown('<div class="section-header">Indexed Supplier Documents</div>', unsafe_allow_html=True)

        try:
            doc_data = list_docs()
            docs = doc_data.get("documents", [])

            # KPI metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown(_kpi("Total Documents", str(len(docs)), "Indexed in vector database", "kpi-blue"), unsafe_allow_html=True)
            with col_m2:
                suppliers_count = len(set(d["supplier_name"] for d in docs))
                st.markdown(_kpi("Suppliers Covered", str(suppliers_count), "Active suppliers with documents", "kpi-green"), unsafe_allow_html=True)
            with col_m3:
                doc_types_count = len(set(d["doc_type"] for d in docs))
                st.markdown(_kpi("Document Types", str(doc_types_count), "SLAs, Contracts, & Policies", "kpi-purple"), unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)

            if docs:
                df_docs = pd.DataFrame(docs)
                df_docs = df_docs.rename(columns={
                    "id": "Doc ID",
                    "filename": "Filename",
                    "supplier_name": "Supplier Name",
                    "doc_type": "Doc Type",
                    "page_count": "Pages",
                    "uploaded_at": "Uploaded At",
                })
                st.dataframe(df_docs, use_container_width=True, hide_index=True)
            else:
                st.info("No supplier documents currently indexed.")
        except APIError as exc:
            _api_error(exc)
