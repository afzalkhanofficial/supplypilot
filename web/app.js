/* ===========================================================================
   SupplyPilot — Frontend Application JavaScript (ES6 async/await)
   =========================================================================== */

// Global State & Chart Handles
let currentRiskChart = null;
let currentForecastChart = null;
let chatHistoryList = [];

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadOverviewData();
    populateProductDropdowns();
});

// ---------------------------------------------------------------------------
// Navigation Router
// ---------------------------------------------------------------------------
function navigateTo(pageId) {
    // Update active navbar item
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => item.classList.remove('active'));
    
    // Find clicked item
    const targetMap = {
        'overview': 0, 'inventory': 1, 'forecast': 2,
        'orders': 3, 'chat': 4, 'rag': 5
    };
    if (navItems[targetMap[pageId]]) {
        navItems[targetMap[pageId]].classList.add('active');
    }

    // Hide all pages, show target page
    const pages = document.querySelectorAll('.page-section');
    pages.forEach(p => p.classList.remove('active'));
    
    const targetPage = document.getElementById(`page-${pageId}`);
    if (targetPage) {
        targetPage.classList.add('active');
    }

    // Page-specific trigger loads
    if (pageId === 'overview') loadOverviewData();
    if (pageId === 'inventory') loadInventoryDetails();
    if (pageId === 'forecast') loadDemandForecast();
    if (pageId === 'orders') loadPendingOrders();
    if (pageId === 'rag') loadRagCatalog();
}

// ---------------------------------------------------------------------------
// Health Probe
// ---------------------------------------------------------------------------
async function checkHealth() {
    try {
        const res = await fetch('/health');
        const data = await res.json();
        
        const apiDot = document.getElementById('api-status-dot');
        const dbDot = document.getElementById('db-status-dot');
        
        if (data.status === 'ok') {
            apiDot.className = 'dot-online';
            apiDot.innerText = '● Online';
        } else {
            apiDot.className = 'dot-offline';
            apiDot.innerText = '● Offline';
        }

        if (data.db_connected) {
            dbDot.className = 'dot-online';
            dbDot.innerText = '● Connected';
        } else {
            dbDot.className = 'dot-offline';
            dbDot.innerText = '● Disconnected';
        }
    } catch (err) {
        document.getElementById('api-status-dot').className = 'dot-offline';
        document.getElementById('api-status-dot').innerText = '● Offline';
        document.getElementById('db-status-dot').className = 'dot-offline';
        document.getElementById('db-status-dot').innerText = '● Disconnected';
    }
}

// Helper: Product Dropdowns Loader
async function populateProductDropdowns() {
    try {
        const res = await fetch('/products');
        const data = await res.json();
        const products = data.products || [];

        const invSelect = document.getElementById('inv-product-select');
        const fcSelect = document.getElementById('fc-product-select');
        const poSelect = document.getElementById('po-product-select');

        const optionsHtml = products.map(p => 
            `<option value="${p.product_id}">Product ${p.product_id} — ${p.product_name}</option>`
        ).join('');

        if (invSelect) invSelect.innerHTML = optionsHtml;
        if (fcSelect) fcSelect.innerHTML = optionsHtml;
        if (poSelect) poSelect.innerHTML = optionsHtml;

    } catch (err) {
        console.error('Failed to load products dropdown:', err);
    }
}

// ---------------------------------------------------------------------------
// PAGE 1 — OVERVIEW
// ---------------------------------------------------------------------------
async function loadOverviewData() {
    try {
        const [scanRes, ordersRes] = await Promise.all([
            fetch('/inventory/scan'),
            fetch('/orders?status=pending&limit=200')
        ]);

        const scanData = await scanRes.json();
        const ordersData = await ordersRes.json();

        // Update KPI Cards
        document.getElementById('ov-total-products').innerText = scanData.scanned || 0;
        document.getElementById('ov-critical-count').innerText = scanData.counts.CRITICAL || 0;
        document.getElementById('ov-warning-count').innerText = scanData.counts.WARNING || 0;
        document.getElementById('ov-pending-orders').innerText = ordersData.total || 0;

        // Render Risk Bar Chart (Chart.js)
        renderRiskDistributionChart(scanData.counts);

        // Render Product Risk Table
        const tbody = document.getElementById('ov-risk-table-body');
        const items = scanData.summary || [];

        if (items.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;">No product inventory data found.</td></tr>';
            return;
        }

        tbody.innerHTML = items.map(item => {
            const badgeClass = item.risk_level === 'CRITICAL' ? 'badge-critical' :
                               item.risk_level === 'WARNING'  ? 'badge-warning' : 'badge-ok';
            return `
                <tr>
                    <td><b>Product ${item.product_id}</b></td>
                    <td><span class="badge ${badgeClass}">${item.risk_level}</span></td>
                    <td>${Number(item.current_stock).toLocaleString()}</td>
                    <td>${Number(item.reorder_point).toLocaleString()}</td>
                    <td>${Number(item.eoq).toLocaleString()}</td>
                    <td>${Number(item.days_of_cover).toFixed(1)}d</td>
                </tr>
            `;
        }).join('');

    } catch (err) {
        console.error('Error loading Overview page data:', err);
    }
}

function renderRiskDistributionChart(counts) {
    const ctx = document.getElementById('chart-risk-distribution').getContext('2d');
    
    if (currentRiskChart) {
        currentRiskChart.destroy();
    }

    currentRiskChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['CRITICAL', 'WARNING', 'OK'],
            datasets: [{
                label: 'Products by Risk Level',
                data: [counts.CRITICAL || 0, counts.WARNING || 0, counts.OK || 0],
                backgroundColor: ['#f43f5e', '#fbbf24', '#34d399'],
                borderColor: ['rgba(244,63,94,0.4)', 'rgba(251,191,36,0.4)', 'rgba(52,211,153,0.4)'],
                borderWidth: 1,
                borderRadius: 8,
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: { backgroundColor: '#0f172a', titleColor: '#f8fafc', bodyColor: '#cbd5e1' }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', precision: 0 } }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// PAGE 2 — INVENTORY
// ---------------------------------------------------------------------------
async function loadInventoryDetails() {
    const select = document.getElementById('inv-product-select');
    if (!select || !select.value) return;

    const pid = select.value;
    try {
        const res = await fetch(`/inventory/${pid}`);
        const inv = await res.json();

        const colorClass = inv.risk_level === 'CRITICAL' ? 'rose' :
                           inv.risk_level === 'WARNING'  ? 'amber' : 'emerald';

        document.getElementById('inv-current-stock').innerText = Number(inv.current_stock).toLocaleString();
        document.getElementById('inv-current-stock').className = `kpi-value ${colorClass}`;

        document.getElementById('inv-days-cover').innerText = `${Number(inv.days_of_cover).toFixed(1)}d`;
        document.getElementById('inv-days-cover').className = `kpi-value ${colorClass}`;
        document.getElementById('inv-risk-sub').innerText = `risk: ${inv.risk_level}`;

        document.getElementById('inv-reorder-point').innerText = Number(inv.reorder_point).toLocaleString();
        document.getElementById('inv-eoq').innerText = Number(inv.eoq).toLocaleString();
        document.getElementById('inv-safety-stock').innerText = Number(inv.safety_stock).toLocaleString();
        document.getElementById('inv-lead-time').innerText = `${inv.lead_time_days}d`;

        // Recommendation Box
        const recBox = document.getElementById('inv-recommendation-box');
        const recTitle = document.getElementById('inv-rec-title');
        const recText = document.getElementById('inv-rec-text');

        recBox.className = `recommendation-banner ${inv.risk_level}`;
        recTitle.className = `rec-title ${inv.risk_level}`;
        recTitle.innerText = `Recommendation: ${inv.action}`;
        recText.innerHTML = `Current stock is <b>${Number(inv.current_stock).toLocaleString()} units</b> vs Reorder Point <b>${Number(inv.reorder_point).toLocaleString()} units</b>. Recommended order size (EOQ): <b>${Number(inv.eoq).toLocaleString()} units</b>.`;

    } catch (err) {
        console.error(`Failed to load inventory for product ${pid}:`, err);
    }
}

// ---------------------------------------------------------------------------
// PAGE 3 — DEMAND FORECAST
// ---------------------------------------------------------------------------
async function loadDemandForecast() {
    const select = document.getElementById('fc-product-select');
    const slider = document.getElementById('fc-days-slider');
    if (!select || !select.value) return;

    const pid = select.value;
    const days = slider ? slider.value : 30;

    try {
        const res = await fetch(`/products/${pid}/forecast?days_ahead=${days}`);
        const fc = await res.json();

        const dates = fc.dates || [];
        const yhat = fc.yhat || [];
        const yhatLower = fc.yhat_lower || [];
        const yhatUpper = fc.yhat_upper || [];

        const totalDemand = fc.total_forecast || 0;
        const dailyAvg = totalDemand / Math.max(dates.length, 1);

        document.getElementById('fc-total-demand').innerText = Math.round(totalDemand).toLocaleString();
        document.getElementById('fc-horizon-sub').innerText = `over next ${days} days`;
        document.getElementById('fc-daily-avg').innerText = Math.round(dailyAvg).toLocaleString();
        document.getElementById('fc-cutoff').innerText = fc.training_end || 'N/A';

        // Render Chart.js Forecast Line & Confidence Interval
        renderForecastChart(dates, yhat, yhatLower, yhatUpper, pid);

    } catch (err) {
        console.error(`Failed to load forecast for product ${pid}:`, err);
    }
}

function renderForecastChart(dates, yhat, yhatLower, yhatUpper, pid) {
    const ctx = document.getElementById('chart-forecast').getContext('2d');

    if (currentForecastChart) {
        currentForecastChart.destroy();
    }

    currentForecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Forecast (yhat)',
                    data: yhat,
                    borderColor: '#38bdf8',
                    borderWidth: 3,
                    pointRadius: 2,
                    tension: 0.2,
                },
                {
                    label: '80% Upper CI',
                    data: yhatUpper,
                    borderColor: 'transparent',
                    pointRadius: 0,
                    tension: 0.2,
                },
                {
                    label: '80% Lower CI',
                    data: yhatLower,
                    borderColor: 'transparent',
                    fill: '-1',
                    backgroundColor: 'rgba(56, 189, 248, 0.12)',
                    pointRadius: 0,
                    tension: 0.2,
                }
            ]
        },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { labels: { color: '#94a3b8' } },
                tooltip: { backgroundColor: '#0f172a', titleColor: '#f8fafc', bodyColor: '#cbd5e1' }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// PAGE 4 — PURCHASE ORDERS
// ---------------------------------------------------------------------------
function switchOrderTab(tabName) {
    const buttons = document.querySelectorAll('#page-orders .tab-btn');
    buttons.forEach(b => b.classList.remove('active'));

    const tabContents = document.querySelectorAll('#page-orders .tab-content');
    tabContents.forEach(c => c.classList.remove('active'));

    if (tabName === 'pending') {
        buttons[0].classList.add('active');
        document.getElementById('ord-tab-pending').classList.add('active');
        loadPendingOrders();
    } else if (tabName === 'all') {
        buttons[1].classList.add('active');
        document.getElementById('ord-tab-all').classList.add('active');
        loadAllOrders();
    } else if (tabName === 'create') {
        buttons[2].classList.add('active');
        document.getElementById('ord-tab-create').classList.add('active');
    }
}

async function loadPendingOrders() {
    const container = document.getElementById('pending-orders-container');
    try {
        const res = await fetch('/orders?status=pending&limit=100');
        const data = await res.json();
        const orders = data.orders || [];

        if (orders.length === 0) {
            container.innerHTML = '<div style="color:#94a3b8; padding:20px; text-align:center;">No pending purchase orders awaiting approval.</div>';
            return;
        }

        container.innerHTML = orders.map(po => `
            <div class="card-panel" style="margin-bottom:14px; border-color: rgba(251,191,36,0.3);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-weight:700; font-size:1.05rem;">Order #${po.id} — Product ${po.product_id}</span>
                    <span class="badge badge-warning">PENDING</span>
                </div>
                <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:8px;">
                    <b>Quantity:</b> ${Number(po.quantity).toLocaleString()} units &nbsp;|&nbsp; 
                    <b>Total Cost:</b> $${Number(po.total_cost).toLocaleString(undefined, {minimumFractionDigits:2})} &nbsp;|&nbsp; 
                    <b>Supplier:</b> ${po.supplier_name}
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; font-style:italic; margin-bottom:14px;">"${po.reason}"</div>
                <div style="display:flex; gap:10px;">
                    <button class="btn btn-success" onclick="handleApproveOrder(${po.id})">✅ Approve</button>
                    <button class="btn btn-danger" onclick="handleRejectOrder(${po.id})">❌ Reject</button>
                </div>
            </div>
        `).join('');

    } catch (err) {
        console.error('Failed to load pending orders:', err);
        container.innerHTML = '<div style="color:#f43f5e;">Failed to load pending orders.</div>';
    }
}

async function handleApproveOrder(orderId) {
    try {
        const res = await fetch(`/orders/${orderId}/status`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'approved' })
        });
        const data = await res.json();
        alert(data.message || 'Order approved successfully!');
        loadPendingOrders();
    } catch (err) {
        alert('Failed to approve order: ' + err);
    }
}

async function handleRejectOrder(orderId) {
    try {
        const res = await fetch(`/orders/${orderId}/status`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'rejected' })
        });
        const data = await res.json();
        alert(data.message || 'Order rejected.');
        loadPendingOrders();
    } catch (err) {
        alert('Failed to reject order: ' + err);
    }
}

async function loadAllOrders() {
    const statusSelect = document.getElementById('ord-filter-status');
    const status = statusSelect ? statusSelect.value : 'All';

    let url = '/orders?limit=200';
    if (status !== 'All') url += `&status=${status}`;

    try {
        const res = await fetch(url);
        const data = await res.json();
        const orders = data.orders || [];

        const tbody = document.getElementById('all-orders-table-body');
        if (orders.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;">No orders found matching criteria.</td></tr>';
            return;
        }

        tbody.innerHTML = orders.map(po => {
            const badgeClass = po.status === 'approved' ? 'badge-ok' :
                               po.status === 'rejected' ? 'badge-critical' : 'badge-warning';
            return `
                <tr>
                    <td><b>#${po.id}</b></td>
                    <td>Product ${po.product_id}</td>
                    <td>${Number(po.quantity).toLocaleString()}</td>
                    <td>${po.supplier_name}</td>
                    <td>$${Number(po.total_cost).toLocaleString(undefined, {minimumFractionDigits:2})}</td>
                    <td><span class="badge ${badgeClass}">${po.status.toUpperCase()}</span></td>
                    <td>${new Date(po.created_at).toLocaleString()}</td>
                </tr>
            `;
        }).join('');

    } catch (err) {
        console.error('Failed to load all orders:', err);
    }
}

async function handleCreateOrder(e) {
    e.preventDefault();
    const pid = document.getElementById('po-product-select').value;
    const qty = document.getElementById('po-quantity-input').value;
    const reason = document.getElementById('po-reason-input').value;

    try {
        const res = await fetch('/orders', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                product_id: parseInt(pid),
                quantity: parseInt(qty),
                reason: reason
            })
        });
        const data = await res.json();
        alert(data.message || 'Purchase order created successfully!');
        switchOrderTab('pending');
    } catch (err) {
        alert('Failed to create purchase order: ' + err);
    }
}

// ---------------------------------------------------------------------------
// PAGE 5 — AGENT CHAT
// ---------------------------------------------------------------------------
function sendAgentQuestion(promptText) {
    document.getElementById('chat-input-field').value = promptText;
    processAgentChat(promptText);
}

function handleAgentFormSubmit(e) {
    e.preventDefault();
    const input = document.getElementById('chat-input-field');
    const question = input.value.trim();
    if (!question) return;

    input.value = '';
    processAgentChat(question);
}

async function processAgentChat(question) {
    const historyContainer = document.getElementById('chat-history');

    // Append User Bubble
    const userBubble = document.createElement('div');
    userBubble.className = 'chat-bubble-user';
    userBubble.innerText = `🧑 ${question}`;
    historyContainer.appendChild(userBubble);

    // Append Agent Thinking Bubble
    const thinkingBubble = document.createElement('div');
    thinkingBubble.className = 'chat-bubble-agent';
    thinkingBubble.innerText = '🤖 SupplyPilot is thinking...';
    historyContainer.appendChild(thinkingBubble);
    historyContainer.scrollTop = historyContainer.scrollHeight;

    try {
        const res = await fetch('/agent/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                chat_history: chatHistoryList
            })
        });
        const data = await res.json();

        const answer = data.answer || 'No response returned.';
        const toolsUsed = data.tools_used || [];
        const steps = data.steps || 1;

        // Update chat history memory
        chatHistoryList.push({ human: question, ai: answer });

        // Replace Thinking Bubble with real Agent response
        const toolPills = toolsUsed.map(t => `<span class="tool-pill">${t}</span>`).join(' ');
        const metaLine = toolPills ? `<div class="chat-meta">Tools: ${toolPills} &nbsp;·&nbsp; Steps: ${steps}</div>` : '';

        thinkingBubble.innerHTML = `🤖 ${answer}${metaLine}`;
        historyContainer.scrollTop = historyContainer.scrollHeight;

    } catch (err) {
        thinkingBubble.innerHTML = `🤖 <span style="color:#f43f5e;">Error connecting to agent: ${err}</span>`;
    }
}

// ---------------------------------------------------------------------------
// PAGE 6 — SUPPLIER INTELLIGENCE (RAG)
// ---------------------------------------------------------------------------
function switchRagTab(tabName) {
    const buttons = document.querySelectorAll('#page-rag .tab-btn');
    buttons.forEach(b => b.classList.remove('active'));

    const tabContents = document.querySelectorAll('#page-rag .tab-content');
    tabContents.forEach(c => c.classList.remove('active'));

    if (tabName === 'search') {
        buttons[0].classList.add('active');
        document.getElementById('rag-tab-search').classList.add('active');
    } else if (tabName === 'upload') {
        buttons[1].classList.add('active');
        document.getElementById('rag-tab-upload').classList.add('active');
    } else if (tabName === 'catalog') {
        buttons[2].classList.add('active');
        document.getElementById('rag-tab-catalog').classList.add('active');
        loadRagCatalog();
    }
}

async function handleRagSearch(e) {
    e.preventDefault();
    const query = document.getElementById('rag-query-input').value.trim();
    const supplier = document.getElementById('rag-supplier-filter').value.trim();
    const docType = document.getElementById('rag-type-filter').value;
    const container = document.getElementById('rag-results-container');

    if (!query) return;

    container.innerHTML = '<div style="color:#94a3b8; padding:16px;">Embedding query & searching vector database...</div>';

    try {
        let url = `/documents/search?q=${encodeURIComponent(query)}&top_k=5`;
        if (supplier) url += `&supplier_name=${encodeURIComponent(supplier)}`;
        if (docType) url += `&doc_type=${encodeURIComponent(docType)}`;

        const res = await fetch(url);
        const data = await res.json();

        if (data.status === 'no_results' || !data.results || data.results.length === 0) {
            container.innerHTML = `<div class="card-panel" style="color:#94a3b8;">${data.message || 'No matching document passages found.'}</div>`;
            return;
        }

        container.innerHTML = data.results.map(r => {
            const simPct = Math.round(r.similarity * 100);
            return `
                <div class="card-panel" style="margin-bottom:14px; border-color: rgba(56,189,248,0.25);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-weight:700; color:#38bdf8; font-size:1.05rem;">📄 ${r.filename} (Rank #${r.rank})</span>
                        <span class="badge badge-ok">${simPct}% Relevance</span>
                    </div>
                    <div style="font-size:0.82rem; color:#94a3b8; margin-bottom:12px;">
                        <b>Supplier:</b> ${r.supplier_name} &nbsp;|&nbsp; <b>Type:</b> ${r.doc_type.toUpperCase()} &nbsp;|&nbsp; <b>Chunk:</b> #${r.chunk_index}
                    </div>
                    <div style="background:rgba(9,13,22,0.9); border-left:4px solid #38bdf8; padding:14px 18px; border-radius:8px; font-size:0.9rem; line-height:1.6; color:#e2e8f0; white-space:pre-wrap;">${r.chunk_text}</div>
                </div>
            `;
        }).join('');

    } catch (err) {
        console.error('Vector search failed:', err);
        container.innerHTML = `<div style="color:#f43f5e;">Search error: ${err}</div>`;
    }
}

async function handleDocumentUpload(e) {
    e.preventDefault();
    const fileInput = document.getElementById('rag-file-input');
    const supplierInput = document.getElementById('rag-supplier-name');
    const docTypeSelect = document.getElementById('rag-doc-type');

    if (!fileInput.files[0] || !supplierInput.value.trim()) {
        alert('Please select a file and enter a supplier name.');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('supplier_name', supplierInput.value.trim());
    formData.append('doc_type', docTypeSelect.value);

    try {
        const res = await fetch('/documents/ingest', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (data.status === 'ok') {
            alert(`Successfully ingested ${data.filename}! Stored ${data.chunks_stored} vector text chunks.`);
            fileInput.value = '';
            switchRagTab('catalog');
        } else {
            alert(`Notice: ${data.message || 'Ingestion returned status ' + data.status}`);
        }
    } catch (err) {
        alert('Document upload failed: ' + err);
    }
}

async function loadRagCatalog() {
    try {
        const res = await fetch('/documents');
        const data = await res.json();
        const docs = data.documents || [];

        // KPI metrics
        document.getElementById('rag-total-docs').innerText = docs.length;
        
        const suppliersSet = new Set(docs.map(d => d.supplier_name));
        document.getElementById('rag-suppliers-count').innerText = suppliersSet.size;

        const typesSet = new Set(docs.map(d => d.doc_type));
        document.getElementById('rag-types-count').innerText = typesSet.size;

        const tbody = document.getElementById('rag-catalog-table-body');
        if (docs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;">No supplier documents currently indexed.</td></tr>';
            return;
        }

        tbody.innerHTML = docs.map(d => `
            <tr>
                <td><b>#${d.id}</b></td>
                <td>📄 ${d.filename}</td>
                <td>${d.supplier_name}</td>
                <td><span class="badge badge-ok">${d.doc_type.toUpperCase()}</span></td>
                <td>${d.page_count || 1}</td>
                <td>${new Date(d.uploaded_at).toLocaleString()}</td>
            </tr>
        `).join('');

    } catch (err) {
        console.error('Failed to load document catalog:', err);
    }
}
