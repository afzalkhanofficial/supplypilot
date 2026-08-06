"""
SupplyPilot Streamlit Dashboard — single-file multi-page app.

Run with:
    streamlit run dashboard/app.py
or:
    python scripts/run_dashboard.py

Pages (sidebar navigation):
  1. Overview        — Fleet KPIs and risk summary table
  2. Inventory       — Per-product stock details and gauges
  3. Demand Forecast — Prophet forecast chart with confidence band
  4. Purchase Orders — Approve / reject orders; create new orders
  5. Agent Chat      — Conversational interface with the AI agent
"""

import sys
from pathlib import Path

# Ensure project root on path so dashboard/api_client.py resolves correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.api_client import (
    APIError,
    agent_chat,
    agent_history,
    approve_order,
    create_order,
    get_forecast,
    get_inventory,
    health,
    ingest_document,
    list_alerts,
    list_documents as list_docs,
    list_orders,
    list_products,
    reject_order,
    scan_inventory,
    search_documents as search_docs,
)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SupplyPilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — dark glassmorphism theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Google font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #111827 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8;
    font-size: 0.9rem;
    padding: 4px 0;
    transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #63b3ed; }

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.95));
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(99,179,237,0.15);
}
.kpi-label  { color: #94a3b8; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px; }
.kpi-value  { font-size: 2.4rem; font-weight: 700; line-height: 1; margin-bottom: 4px; }
.kpi-sub    { color: #64748b; font-size: 0.75rem; }
.kpi-blue   { color: #63b3ed; }
.kpi-amber  { color: #f6ad55; }
.kpi-red    { color: #fc8181; }
.kpi-green  { color: #68d391; }

/* ── Risk badge ── */
.badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:0.72rem; font-weight:600; letter-spacing:0.05em; }
.badge-critical { background:rgba(252,129,129,0.15); color:#fc8181; border:1px solid rgba(252,129,129,0.3); }
.badge-warning  { background:rgba(246,173,85,0.15);  color:#f6ad55; border:1px solid rgba(246,173,85,0.3); }
.badge-ok       { background:rgba(104,211,145,0.15); color:#68d391; border:1px solid rgba(104,211,145,0.3); }

/* ── Section header ── */
.section-header {
    font-size: 1.1rem; font-weight: 600; color: #e2e8f0;
    border-left: 3px solid #63b3ed;
    padding-left: 10px; margin: 20px 0 14px 0;
}

/* ── Chat bubbles ── */
.chat-user {
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 12px 12px 2px 12px;
    padding: 12px 16px; margin: 8px 0; color: #e2e8f0;
}
.chat-agent {
    background: rgba(30,41,59,0.7);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px 12px 12px 2px;
    padding: 12px 16px; margin: 8px 0; color: #cbd5e0;
}
.tool-pill {
    display:inline-block; margin:2px 3px;
    background:rgba(99,179,237,0.1); color:#63b3ed;
    border:1px solid rgba(99,179,237,0.25);
    border-radius:999px; padding:2px 9px; font-size:0.7rem; font-weight:500;
}
.agent-meta { color:#64748b; font-size:0.72rem; margin-top:8px; }

/* ── Inputs ── */
.stTextInput input, .stSelectbox select, .stNumberInput input, .stTextArea textarea {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white; border: none; border-radius: 8px;
    font-weight: 500; padding: 8px 20px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(37,99,235,0.4);
}

/* ── Divider ── */
hr { border-color: rgba(99,179,237,0.1) !important; }

/* ── Page title ── */
.page-title {
    font-size: 1.8rem; font-weight: 700; color: #f1f5f9;
    margin-bottom: 4px;
}
.page-sub { color: #64748b; font-size: 0.9rem; margin-bottom: 24px; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly dark layout defaults
# ---------------------------------------------------------------------------

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(family="Inter", color="#94a3b8"),
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
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {color_class}">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def _api_error(exc: APIError) -> None:
    st.error(f"**API Error:** {exc}")
    if "port 8000" in str(exc):
        st.info("Start the API with: `python scripts/run_api.py`")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 24px 0;">
        <div style="font-size:2rem;">🚀</div>
        <div style="font-size:1.2rem; font-weight:700; color:#e2e8f0;">SupplyPilot</div>
        <div style="font-size:0.72rem; color:#475569; letter-spacing:0.1em;">SUPPLY CHAIN AI</div>
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

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Live health indicator
    try:
        h = health()
        db_ok = h.get("db_connected", False)
        st.markdown(
            f'<div style="font-size:0.75rem; color:#64748b;">API <span style="color:{"#68d391" if db_ok else "#fc8181"}">●</span> {"Online" if db_ok else "Offline"} &nbsp;|&nbsp; DB <span style="color:{"#68d391" if db_ok else "#fc8181"}">●</span> {"Connected" if db_ok else "Disconnected"}</div>',
            unsafe_allow_html=True,
        )
    except APIError:
        st.markdown('<div style="font-size:0.75rem; color:#fc8181;">● API Offline</div>', unsafe_allow_html=True)


# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================

if page == "📊  Overview":
    st.markdown('<div class="page-title">Fleet Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Real-time inventory health across all products</div>', unsafe_allow_html=True)

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

        # Risk distribution bar
        st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
        total = scan["scanned"] or 1
        fig_dist = go.Figure(go.Bar(
            x=["CRITICAL", "WARNING", "OK"],
            y=[counts["CRITICAL"], counts["WARNING"], counts["OK"]],
            marker_color=["#fc8181", "#f6ad55", "#68d391"],
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
                "product_id": "Product",
                "risk_badge": "Risk",
                "current_stock": "Stock",
                "reorder_point": "Reorder Point",
                "eoq": "EOQ",
                "days_of_cover": "Cover",
            })
            st.write(display_df.to_html(escape=False, index=False, classes=""), unsafe_allow_html=True)

    except APIError as exc:
        _api_error(exc)


# ===========================================================================
# PAGE 2 — INVENTORY
# ===========================================================================

elif page == "📦  Inventory":
    st.markdown('<div class="page-title">Inventory Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Per-product stock levels, reorder points, and EOQ</div>', unsafe_allow_html=True)

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

        # Stock vs. thresholds bar chart
        st.markdown('<div class="section-header">Stock vs. Thresholds</div>', unsafe_allow_html=True)
        fig = go.Figure()
        categories = ["Current Stock", "Reorder Point", "Safety Stock", "EOQ"]
        values = [inv["current_stock"], inv["reorder_point"], inv["safety_stock"], inv["eoq"]]
        colors = [
            "#fc8181" if risk == "CRITICAL" else "#f6ad55" if risk == "WARNING" else "#68d391",
            "#f6ad55", "#63b3ed", "#a78bfa",
        ]
        fig.add_trace(go.Bar(x=categories, y=values, marker_color=colors, text=[f"{v:,.0f}" for v in values], textposition="auto"))
        fig.update_layout(**_PLOTLY_LAYOUT, title=f"Product {pid} — Inventory Snapshot", height=360)
        st.plotly_chart(fig, use_container_width=True)

        # Action recommendation
        st.markdown('<div class="section-header">Recommended Action</div>', unsafe_allow_html=True)
        badge_html = _badge(risk)
        st.markdown(
            f'<div style="background:rgba(30,41,59,0.7);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:16px 20px;">'
            f'{badge_html}&nbsp;&nbsp;<span style="color:#cbd5e0;">{inv["action"]}</span></div>',
            unsafe_allow_html=True,
        )

    except APIError as exc:
        _api_error(exc)


# ===========================================================================
# PAGE 3 — DEMAND FORECAST
# ===========================================================================

elif page == "📈  Demand Forecast":
    st.markdown('<div class="page-title">Demand Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Prophet model predictions with 80% confidence interval</div>', unsafe_allow_html=True)

    try:
        products = list_products()
        product_map = {f"Product {p['product_id']} — {p['product_name']}": p["product_id"] for p in products}

        col_sel, col_days = st.columns([3, 1])
        with col_sel:
            selected_label = st.selectbox("Select product", list(product_map.keys()))
        with col_days:
            days_ahead = st.slider("Days ahead", min_value=7, max_value=90, value=30, step=7)

        pid = product_map[selected_label]

        with st.spinner("Loading forecast..."):
            fc = get_forecast(pid, days_ahead)

        total_fc = sum(fc["yhat"])
        daily_avg = total_fc / days_ahead

        st.markdown("<br/>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(_kpi("Total Forecast", f"{total_fc:,.0f}", f"over {days_ahead} days", "kpi-blue"), unsafe_allow_html=True)
        c2.markdown(_kpi("Daily Average", f"{daily_avg:,.0f}", "units/day", "kpi-blue"), unsafe_allow_html=True)
        c3.markdown(_kpi("Model Cut-off", fc["training_end"], "last training date", "kpi-blue"), unsafe_allow_html=True)

        st.markdown('<div class="section-header">Forecast Chart</div>', unsafe_allow_html=True)

        fig = go.Figure()

        # Confidence band
        fig.add_trace(go.Scatter(
            x=fc["dates"] + fc["dates"][::-1],
            y=fc["yhat_upper"] + fc["yhat_lower"][::-1],
            fill="toself",
            fillcolor="rgba(99,179,237,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% CI",
            hoverinfo="skip",
        ))

        # Point forecast
        fig.add_trace(go.Scatter(
            x=fc["dates"], y=fc["yhat"],
            mode="lines+markers",
            line=dict(color="#63b3ed", width=2.5),
            marker=dict(size=5, color="#63b3ed"),
            name="Forecast (yhat)",
        ))

        fig.update_layout(
            **_PLOTLY_LAYOUT,
            title=f"Product {pid} — {days_ahead}-Day Demand Forecast",
            height=420,
            xaxis_title="Date",
            yaxis_title="Units",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Data table toggle
        with st.expander("View raw forecast data"):
            df_fc = pd.DataFrame({
                "Date": fc["dates"],
                "Forecast (yhat)": [f"{v:,.0f}" for v in fc["yhat"]],
                "Lower (80% CI)": [f"{v:,.0f}" for v in fc["yhat_lower"]],
                "Upper (80% CI)": [f"{v:,.0f}" for v in fc["yhat_upper"]],
            })
            st.dataframe(df_fc, use_container_width=True, hide_index=True)

    except APIError as exc:
        _api_error(exc)


# ===========================================================================
# PAGE 4 — PURCHASE ORDERS
# ===========================================================================

elif page == "🛒  Purchase Orders":
    st.markdown('<div class="page-title">Purchase Orders</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Review pending orders and create new replenishment orders</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["⏳  Pending", "📋  All Orders", "➕  Create Order"])

    # ── Tab 1: Pending ────────────────────────────────────────────────────
    with tab1:
        try:
            data = list_orders(status="pending")
            orders = data["orders"]
            if not orders:
                st.info("No pending orders — the queue is clear.")
            else:
                st.markdown(f'<div class="section-header">{len(orders)} Pending Order(s)</div>', unsafe_allow_html=True)
                for order in orders:
                    with st.container():
                        st.markdown(
                            f'<div style="background:rgba(30,41,59,0.7);border:1px solid rgba(255,255,255,0.08);'
                            f'border-radius:12px;padding:16px 20px;margin-bottom:12px;">'
                            f'<b style="color:#e2e8f0;">Order #{order["id"]}</b> &nbsp;·&nbsp; '
                            f'Product {order["product_id"]} &nbsp;·&nbsp; '
                            f'<span style="color:#63b3ed;">{order["quantity"]:,} units</span> &nbsp;·&nbsp; '
                            f'<span style="color:#f6ad55;">${order["estimated_cost"]:,.2f}</span> &nbsp;·&nbsp; '
                            f'{order["supplier_name"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if order.get("agent_reasoning"):
                            with st.expander("Agent reasoning"):
                                st.write(order["agent_reasoning"])
                        ca, cb, _ = st.columns([1, 1, 5])
                        if ca.button("✅ Approve", key=f"approve_{order['id']}"):
                            try:
                                approve_order(order["id"])
                                st.success(f"Order #{order['id']} approved.")
                                st.rerun()
                            except APIError as e:
                                st.error(str(e))
                        if cb.button("❌ Reject", key=f"reject_{order['id']}"):
                            try:
                                reject_order(order["id"])
                                st.warning(f"Order #{order['id']} rejected.")
                                st.rerun()
                            except APIError as e:
                                st.error(str(e))
        except APIError as exc:
            _api_error(exc)

    # ── Tab 2: All Orders ─────────────────────────────────────────────────
    with tab2:
        try:
            status_filter = st.selectbox("Filter by status", ["All", "pending", "approved", "rejected"], key="order_filter")
            data = list_orders(status=None if status_filter == "All" else status_filter)
            orders = data["orders"]
            if not orders:
                st.info("No orders found.")
            else:
                df_orders = pd.DataFrame(orders)
                df_orders["created_at"] = pd.to_datetime(df_orders["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
                df_orders["estimated_cost"] = df_orders["estimated_cost"].apply(lambda x: f"${x:,.2f}")
                df_orders["quantity"] = df_orders["quantity"].apply(lambda x: f"{x:,}")
                st.dataframe(
                    df_orders[["id", "product_id", "quantity", "supplier_name", "estimated_cost", "status", "created_at"]].rename(columns={
                        "id": "ID", "product_id": "Product", "quantity": "Qty",
                        "supplier_name": "Supplier", "estimated_cost": "Cost",
                        "status": "Status", "created_at": "Created",
                    }),
                    use_container_width=True, hide_index=True,
                )
        except APIError as exc:
            _api_error(exc)

    # ── Tab 3: Create Order ───────────────────────────────────────────────
    with tab3:
        try:
            products = list_products()
            product_map = {f"Product {p['product_id']} — {p['product_name']}": p["product_id"] for p in products}

            st.markdown('<div class="section-header">New Purchase Order</div>', unsafe_allow_html=True)
            with st.form("create_order_form"):
                sel_label = st.selectbox("Product", list(product_map.keys()))
                quantity = st.number_input("Quantity (units)", min_value=1, value=100, step=10)
                reason = st.text_area("Reason", placeholder="e.g. Stock below reorder point, 7 days of cover remaining.")
                submitted = st.form_submit_button("Create Order")

            if submitted:
                if not reason.strip():
                    st.error("Please provide a reason for the order.")
                else:
                    pid = product_map[sel_label]
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
        st.session_state.chat_messages = []   # list of {"role": "user"|"assistant", "content": str, "meta": dict}

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

        # Build chat_history for multi-turn context (pairs of user/ai turns)
        history_pairs = []
        msgs = st.session_state.chat_messages[:-1]  # exclude the just-added user msg
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
                                <div style="background:rgba(30,41,59,0.8); border:1px solid rgba(99,179,237,0.25); border-radius:12px; padding:16px; margin-bottom:12px;">
                                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                        <span style="font-weight:600; color:#63b3ed; font-size:1.05rem;">
                                            📄 {res['filename']} (Rank #{res['rank']})
                                        </span>
                                        <span style="background:rgba(104,211,145,0.15); color:#68d391; border:1px solid rgba(104,211,145,0.3); border-radius:12px; padding:2px 10px; font-size:0.8rem; font-weight:600;">
                                            {sim_pct}% Relevance
                                        </span>
                                    </div>
                                    <div style="font-size:0.82rem; color:#94a3b8; margin-bottom:10px;">
                                        <b>Supplier:</b> {res['supplier_name']} &nbsp;|&nbsp; <b>Type:</b> {res['doc_type'].upper()} &nbsp;|&nbsp; <b>Chunk:</b> #{res['chunk_index']}
                                    </div>
                                    <div style="background:rgba(15,23,42,0.9); border-left:3px solid #63b3ed; padding:12px 16px; border-radius:4px; font-size:0.9rem; line-height:1.5; color:#e2e8f0; white-space:pre-wrap;">{res['chunk_text']}</div>
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

