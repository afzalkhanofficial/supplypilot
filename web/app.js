/* ===========================================================================
   SupplyPilot — Frontend Application JavaScript (ES6 async/await)
   Enterprise Dark Theme Compatible (Tailwind CSS)
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
    setInterval(checkHealth, 10000); // Polling every 10 seconds for real-time live status
    loadOverviewData();
    populateProductDropdowns();
});


// ---------------------------------------------------------------------------
// Navigation Router
// ---------------------------------------------------------------------------
function navigateTo(pageId) {
    // Update sidebar navigation buttons
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.classList.remove('active', 'bg-brand-purple/15', 'text-white', 'border-brand-purple/40', 'shadow-md');
        item.classList.add('text-gray-400', 'hover:text-white', 'hover:bg-surface-dark', 'border-transparent');
    });

    const targetMap = {
        'overview': 0, 'inventory': 1, 'forecast': 2,
        'orders': 3, 'chat': 4, 'rag': 5
    };
    const activeIdx = targetMap[pageId];
    if (activeIdx !== undefined && navItems[activeIdx]) {
        const item = navItems[activeIdx];
        item.classList.add('active', 'bg-brand-purple/15', 'text-white', 'border-brand-purple/40', 'shadow-md');
        item.classList.remove('text-gray-400', 'hover:text-white', 'hover:bg-surface-dark', 'border-transparent');
    }

    // Hide all page sections, show target page
    const pages = document.querySelectorAll('.page-section');
    pages.forEach(p => p.classList.remove('active'));

    const targetPage = document.getElementById(`page-${pageId}`);
    if (targetPage) {
        targetPage.classList.add('active');
    }

    // Trigger page-specific data fetch
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
            apiDot.innerHTML = '<span class="w-2 h-2 rounded-full bg-brand-emerald animate-pulse mr-1.5"></span>Online';
            apiDot.className = 'flex items-center text-brand-emerald font-semibold';
        } else {
            apiDot.innerHTML = '<span class="w-2 h-2 rounded-full bg-brand-rose mr-1.5"></span>Offline';
            apiDot.className = 'flex items-center text-brand-rose font-semibold';
        }

        if (data.db_connected) {
            dbDot.innerHTML = '<span class="w-2 h-2 rounded-full bg-brand-emerald animate-pulse mr-1.5"></span>Connected';
            dbDot.className = 'flex items-center text-brand-emerald font-semibold';
        } else {
            dbDot.innerHTML = '<span class="w-2 h-2 rounded-full bg-brand-rose mr-1.5"></span>Disconnected';
            dbDot.className = 'flex items-center text-brand-rose font-semibold';
        }
    } catch (err) {
        console.error('Health probe error:', err);
    }
}

// Helper: Populate Product Dropdowns
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
// PAGE 1 — FLEET OVERVIEW
// ---------------------------------------------------------------------------
async function loadOverviewData() {
    try {
        const [scanRes, ordersRes] = await Promise.all([
            fetch('/inventory/scan'),
            fetch('/orders?status=pending&limit=200')
        ]);

        const scanData = await scanRes.json();
        const ordersData = await ordersRes.json();

        // Update KPI Stats
        document.getElementById('ov-total-products').innerText = scanData.scanned || 0;
        document.getElementById('ov-critical-count').innerText = scanData.counts.CRITICAL || 0;
        document.getElementById('ov-warning-count').innerText = scanData.counts.WARNING || 0;
        document.getElementById('ov-pending-orders').innerText = ordersData.total || 0;

        // Render Bar Chart
        renderRiskDistributionChart(scanData.counts);

        // Render Product Matrix Table
        const tbody = document.getElementById('ov-risk-table-body');
        const items = scanData.summary || [];

        if (items.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="py-6 text-center text-gray-500">No product inventory data found.</td></tr>';
            return;
        }

        tbody.innerHTML = items.map(item => {
            const badgeClass = item.risk_level === 'CRITICAL' ? 'bg-brand-rose/10 text-brand-rose border-brand-rose/30' :
                               item.risk_level === 'WARNING'  ? 'bg-brand-amber/10 text-brand-amber border-brand-amber/30' : 
                                                                'bg-brand-emerald/10 text-brand-emerald border-brand-emerald/30';
            return `
                <tr class="hover:bg-surface-card/60 transition-colors">
                    <td class="py-3.5 px-6 font-semibold text-white">Product ${item.product_id}</td>
                    <td class="py-3.5 px-6"><span class="px-2.5 py-1 text-[10px] font-mono uppercase font-bold rounded-full border ${badgeClass}">${item.risk_level}</span></td>
                    <td class="py-3.5 px-6 font-mono">${Number(item.current_stock).toLocaleString()}</td>
                    <td class="py-3.5 px-6 font-mono text-gray-400">${Number(item.reorder_point).toLocaleString()}</td>
                    <td class="py-3.5 px-6 font-mono text-gray-400">${Number(item.eoq).toLocaleString()}</td>
                    <td class="py-3.5 px-6 font-mono font-bold ${item.days_of_cover < 7 ? 'text-brand-rose' : 'text-brand-cyan'}">${Number(item.days_of_cover).toFixed(1)}d</td>
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
                label: 'Products',
                data: [counts.CRITICAL || 0, counts.WARNING || 0, counts.OK || 0],
                backgroundColor: ['#EF4444', '#F59E0B', '#10B981'],
                borderColor: ['rgba(239,68,68,0.3)', 'rgba(245,158,11,0.3)', 'rgba(16,185,129,0.3)'],
                borderWidth: 1,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { backgroundColor: '#161b22', titleColor: '#f3f4f6', bodyColor: '#9ca3af' }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af', font: { family: 'JetBrains Mono' } } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af', precision: 0, font: { family: 'JetBrains Mono' } } }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// PAGE 2 — INVENTORY STATUS
// ---------------------------------------------------------------------------
async function loadInventoryDetails() {
    const select = document.getElementById('inv-product-select');
    if (!select || !select.value) return;

    const pid = select.value;
    try {
        const res = await fetch(`/inventory/${pid}`);
        const inv = await res.json();

        const colorClass = inv.risk_level === 'CRITICAL' ? 'text-brand-rose' :
                           inv.risk_level === 'WARNING'  ? 'text-brand-amber' : 'text-brand-emerald';

        document.getElementById('inv-current-stock').innerText = Number(inv.current_stock).toLocaleString();
        document.getElementById('inv-current-stock').className = `text-3xl font-bold font-mono mt-1 ${colorClass}`;

        document.getElementById('inv-days-cover').innerText = `${Number(inv.days_of_cover).toFixed(1)}d`;
        document.getElementById('inv-days-cover').className = `text-3xl font-bold font-mono mt-1 ${colorClass}`;
        document.getElementById('inv-risk-sub').innerText = `risk level: ${inv.risk_level}`;

        document.getElementById('inv-reorder-point').innerText = Number(inv.reorder_point).toLocaleString();
        document.getElementById('inv-eoq').innerText = Number(inv.eoq).toLocaleString();
        document.getElementById('inv-safety-stock').innerText = Number(inv.safety_stock).toLocaleString();
        document.getElementById('inv-lead-time').innerText = `${inv.lead_time_days} days`;

        // Action Recommendation Box
        const recBox = document.getElementById('inv-recommendation-box');
        const recTitle = document.getElementById('inv-rec-title');
        const recText = document.getElementById('inv-rec-text');

        const borderClass = inv.risk_level === 'CRITICAL' ? 'border-l-brand-rose' :
                            inv.risk_level === 'WARNING'  ? 'border-l-brand-amber' : 'border-l-brand-emerald';
        const titleClass  = inv.risk_level === 'CRITICAL' ? 'text-brand-rose' :
                            inv.risk_level === 'WARNING'  ? 'text-brand-amber' : 'text-brand-emerald';

        recBox.className = `bg-surface-dark border border-gray-800 p-6 rounded-xl border-l-4 ${borderClass}`;
        recTitle.className = `font-bold text-lg mb-2 ${titleClass}`;
        recTitle.innerText = `Recommendation: ${inv.action}`;
        recText.innerHTML = `Current stock level is <b>${Number(inv.current_stock).toLocaleString()} units</b> against Reorder Point <b>${Number(inv.reorder_point).toLocaleString()} units</b>. Recommended order batch size (EOQ): <b>${Number(inv.eoq).toLocaleString()} units</b>.`;

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

        // Render Chart.js Forecast Line & Confidence Interval Shading
        renderForecastChart(dates, yhat, yhatLower, yhatUpper);

    } catch (err) {
        console.error(`Failed to load forecast for product ${pid}:`, err);
    }
}

function renderForecastChart(dates, yhat, yhatLower, yhatUpper) {
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
                    label: 'Prophet Forecast (yhat)',
                    data: yhat,
                    borderColor: '#8C4FFF',
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
                    backgroundColor: 'rgba(140, 79, 255, 0.15)',
                    pointRadius: 0,
                    tension: 0.2,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { labels: { color: '#9ca3af', font: { family: 'JetBrains Mono', size: 11 } } },
                tooltip: { backgroundColor: '#161b22', titleColor: '#f3f4f6', bodyColor: '#9ca3af' }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af', font: { family: 'JetBrains Mono', size: 10 } } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af', font: { family: 'JetBrains Mono', size: 10 } } }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// PAGE 4 — PURCHASE ORDERS
// ---------------------------------------------------------------------------
function switchOrderTab(tabName) {
    const buttons = document.querySelectorAll('.ord-tab-btn');
    buttons.forEach(b => {
        b.classList.remove('active', 'bg-brand-purple', 'text-white');
        b.classList.add('text-gray-400', 'hover:text-white', 'hover:bg-surface-dark');
    });

    const tabContents = document.querySelectorAll('#page-orders .tab-content');
    tabContents.forEach(c => c.classList.remove('active'));

    const tabMap = { 'pending': 0, 'all': 1, 'create': 2 };
    const idx = tabMap[tabName];

    if (idx !== undefined && buttons[idx]) {
        buttons[idx].classList.add('active', 'bg-brand-purple', 'text-white');
        buttons[idx].classList.remove('text-gray-400', 'hover:text-white', 'hover:bg-surface-dark');
    }

    const targetContent = document.getElementById(`ord-tab-${tabName}`);
    if (targetContent) {
        targetContent.classList.add('active');
    }

    if (tabName === 'pending') loadPendingOrders();
    if (tabName === 'all') loadAllOrders();
}

async function loadPendingOrders() {
    const container = document.getElementById('pending-orders-container');
    try {
        const res = await fetch('/orders?status=pending&limit=100');
        const data = await res.json();
        const orders = data.orders || [];

        if (orders.length === 0) {
            container.innerHTML = '<div class="bg-surface-dark border border-gray-800 p-8 rounded-xl text-center text-gray-500">No pending purchase orders awaiting approval.</div>';
            return;
        }

        container.innerHTML = orders.map(po => `
            <div class="bg-surface-dark border border-gray-800 p-5 rounded-xl border-l-4 border-l-brand-amber space-y-3">
                <div class="flex justify-between items-center">
                    <span class="font-bold text-white text-base">Purchase Order #${po.id} — Product ${po.product_id}</span>
                    <span class="px-2.5 py-0.5 text-[10px] font-mono uppercase font-bold rounded-full bg-brand-amber/10 text-brand-amber border border-brand-amber/30">PENDING</span>
                </div>
                <div class="text-xs text-gray-400 font-mono space-x-4">
                    <span><b>Qty:</b> ${Number(po.quantity).toLocaleString()} units</span>
                    <span>•</span>
                    <span><b>Total Cost:</b> $${Number(po.total_cost).toLocaleString(undefined, {minimumFractionDigits:2})}</span>
                    <span>•</span>
                    <span><b>Supplier:</b> ${po.supplier_name}</span>
                </div>
                <div class="text-xs text-gray-300 italic bg-black/40 p-3 rounded-lg border border-gray-800">"${po.reason}"</div>
                <div class="flex space-x-3 pt-1">
                    <button onclick="handleApproveOrder(${po.id})" class="px-4 py-2 bg-brand-emerald hover:bg-emerald-600 text-white text-xs font-semibold rounded-lg transition-colors shadow-md">✅ Approve Order</button>
                    <button onclick="handleRejectOrder(${po.id})" class="px-4 py-2 bg-brand-rose hover:bg-rose-600 text-white text-xs font-semibold rounded-lg transition-colors shadow-md">❌ Reject Order</button>
                </div>
            </div>
        `).join('');

    } catch (err) {
        console.error('Failed to load pending orders:', err);
        container.innerHTML = '<div class="text-brand-rose">Failed to load pending orders.</div>';
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
            tbody.innerHTML = '<tr><td colspan="7" class="py-6 text-center text-gray-500">No purchase orders matching criteria.</td></tr>';
            return;
        }

        tbody.innerHTML = orders.map(po => {
            const badgeClass = po.status === 'approved' ? 'bg-brand-emerald/10 text-brand-emerald border-brand-emerald/30' :
                               po.status === 'rejected' ? 'bg-brand-rose/10 text-brand-rose border-brand-rose/30' : 
                                                          'bg-brand-amber/10 text-brand-amber border-brand-amber/30';
            return `
                <tr class="hover:bg-surface-card/60 transition-colors">
                    <td class="py-3.5 px-6 font-mono font-bold text-white">#${po.id}</td>
                    <td class="py-3.5 px-6 font-semibold text-gray-200">Product ${po.product_id}</td>
                    <td class="py-3.5 px-6 font-mono">${Number(po.quantity).toLocaleString()}</td>
                    <td class="py-3.5 px-6 text-gray-300">${po.supplier_name}</td>
                    <td class="py-3.5 px-6 font-mono">$${Number(po.total_cost).toLocaleString(undefined, {minimumFractionDigits:2})}</td>
                    <td class="py-3.5 px-6"><span class="px-2.5 py-1 text-[10px] font-mono uppercase font-bold rounded-full border ${badgeClass}">${po.status.toUpperCase()}</span></td>
                    <td class="py-3.5 px-6 font-mono text-gray-500">${new Date(po.created_at).toLocaleString()}</td>
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

    // Append User Message Bubble
    const userBubble = document.createElement('div');
    userBubble.className = 'bg-brand-purple/15 border border-brand-purple/30 p-4 rounded-xl text-sm leading-relaxed text-white self-end ml-12';
    userBubble.innerHTML = `<span class="text-brand-purple font-semibold">🧑 You:</span> ${question}`;
    historyContainer.appendChild(userBubble);

    // Append Agent Thinking Bubble
    const thinkingBubble = document.createElement('div');
    thinkingBubble.className = 'bg-surface-card border border-gray-800/80 p-4 rounded-xl text-sm leading-relaxed text-gray-300 mr-12';
    thinkingBubble.innerHTML = `<span class="text-brand-purple font-semibold">🤖 SupplyPilot:</span> <span class="animate-pulse">Reasoning over tools & data...</span>`;
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

        // Tool badges
        const toolPills = toolsUsed.map(t => `<span class="font-mono text-[10px] bg-brand-cyan/10 text-brand-cyan border border-brand-cyan/20 px-2 py-0.5 rounded-full mr-1">${t}</span>`).join('');
        const metaLine = toolPills ? `<div class="mt-3 pt-2 border-t border-gray-800 text-xs text-gray-500 font-mono flex items-center gap-2"><span>Tools Executed:</span> ${toolPills} <span class="ml-auto">Steps: ${steps}</span></div>` : '';

        thinkingBubble.innerHTML = `<span class="text-brand-purple font-semibold">🤖 SupplyPilot:</span> ${answer}${metaLine}`;
        historyContainer.scrollTop = historyContainer.scrollHeight;

    } catch (err) {
        thinkingBubble.innerHTML = `<span class="text-brand-purple font-semibold">🤖 SupplyPilot:</span> <span class="text-brand-rose">Error connecting to agent backend: ${err}</span>`;
    }
}

// ---------------------------------------------------------------------------
// PAGE 6 — SUPPLIER INTELLIGENCE (RAG)
// ---------------------------------------------------------------------------
function switchRagTab(tabName) {
    const buttons = document.querySelectorAll('.rag-tab-btn');
    buttons.forEach(b => {
        b.classList.remove('active', 'bg-brand-purple', 'text-white');
        b.classList.add('text-gray-400', 'hover:text-white', 'hover:bg-surface-dark');
    });

    const tabContents = document.querySelectorAll('#page-rag .tab-content');
    tabContents.forEach(c => c.classList.remove('active'));

    const tabMap = { 'search': 0, 'upload': 1, 'catalog': 2 };
    const idx = tabMap[tabName];

    if (idx !== undefined && buttons[idx]) {
        buttons[idx].classList.add('active', 'bg-brand-purple', 'text-white');
        buttons[idx].classList.remove('text-gray-400', 'hover:text-white', 'hover:bg-surface-dark');
    }

    const targetContent = document.getElementById(`rag-tab-${tabName}`);
    if (targetContent) {
        targetContent.classList.add('active');
    }

    if (tabName === 'catalog') loadRagCatalog();
}

async function handleRagSearch(e) {
    e.preventDefault();
    const query = document.getElementById('rag-query-input').value.trim();
    const supplier = document.getElementById('rag-supplier-filter').value.trim();
    const docType = document.getElementById('rag-type-filter').value;
    const container = document.getElementById('rag-results-container');

    if (!query) return;

    container.innerHTML = '<div class="p-4 text-xs font-mono text-gray-400">Embedding query & searching vector database...</div>';

    try {
        let url = `/documents/search?q=${encodeURIComponent(query)}&top_k=5`;
        if (supplier) url += `&supplier_name=${encodeURIComponent(supplier)}`;
        if (docType) url += `&doc_type=${encodeURIComponent(docType)}`;

        const res = await fetch(url);
        const data = await res.json();

        if (data.status === 'no_results' || !data.results || data.results.length === 0) {
            container.innerHTML = `<div class="bg-surface-dark border border-gray-800 p-5 rounded-xl text-xs text-gray-500">${data.message || 'No matching document passages found.'}</div>`;
            return;
        }

        container.innerHTML = data.results.map(r => {
            const simPct = Math.round(r.similarity * 100);
            return `
                <div class="bg-surface-dark border border-gray-800 p-5 rounded-xl border-l-4 border-l-brand-cyan space-y-2">
                    <div class="flex justify-between items-center">
                        <span class="font-bold text-brand-cyan text-sm">📄 ${r.filename} <span class="text-xs text-gray-500 font-mono">(Rank #${r.rank})</span></span>
                        <span class="px-2.5 py-0.5 text-[10px] font-mono font-bold rounded-full bg-brand-emerald/10 text-brand-emerald border border-brand-emerald/30">${simPct}% Relevance</span>
                    </div>
                    <div class="text-xs text-gray-400 font-mono space-x-3">
                        <span><b>Supplier:</b> ${r.supplier_name}</span>
                        <span>•</span>
                        <span><b>Type:</b> ${r.doc_type.toUpperCase()}</span>
                        <span>•</span>
                        <span><b>Chunk:</b> #${r.chunk_index}</span>
                    </div>
                    <div class="bg-black/60 border-l-2 border-brand-cyan p-3 rounded-lg text-xs leading-relaxed text-gray-200 font-mono whitespace-pre-wrap mt-2">${r.chunk_text}</div>
                </div>
            `;
        }).join('');

    } catch (err) {
        console.error('Vector search failed:', err);
        container.innerHTML = `<div class="text-brand-rose text-xs">Search error: ${err}</div>`;
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

        document.getElementById('rag-total-docs').innerText = docs.length;

        const suppliersSet = new Set(docs.map(d => d.supplier_name));
        document.getElementById('rag-suppliers-count').innerText = suppliersSet.size;

        const typesSet = new Set(docs.map(d => d.doc_type));
        document.getElementById('rag-types-count').innerText = typesSet.size;

        const tbody = document.getElementById('rag-catalog-table-body');
        if (docs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="py-6 text-center text-gray-500">No supplier documents currently indexed.</td></tr>';
            return;
        }

        tbody.innerHTML = docs.map(d => `
            <tr class="hover:bg-surface-card/60 transition-colors">
                <td class="py-3.5 px-6 font-mono font-bold text-white">#${d.id}</td>
                <td class="py-3.5 px-6 font-semibold text-gray-200">📄 ${d.filename}</td>
                <td class="py-3.5 px-6 text-gray-300">${d.supplier_name}</td>
                <td class="py-3.5 px-6"><span class="px-2.5 py-1 text-[10px] font-mono uppercase font-bold rounded-full bg-brand-emerald/10 text-brand-emerald border border-brand-emerald/30">${d.doc_type.toUpperCase()}</span></td>
                <td class="py-3.5 px-6 font-mono text-gray-400">${d.page_count || 1}</td>
                <td class="py-3.5 px-6 font-mono text-gray-500">${new Date(d.uploaded_at).toLocaleString()}</td>
            </tr>
        `).join('');

    } catch (err) {
        console.error('Failed to load document catalog:', err);
    }
}
