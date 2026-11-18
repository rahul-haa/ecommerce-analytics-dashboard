# ⚡ Nomad Retail: E-Commerce Business Analytics Dashboard

An enterprise-grade, modern, and professional **E-Commerce Business Analytics Platform** designed to convert raw transactional data into high-conviction business decisions.

Faithfully modeled after the **Stripe Dashboard / Nomad List** dark theme aesthetic, this platform delivers executive-level visibility into revenue velocity, customer lifetime value, catalog profitability, and machine-learning-driven customer segmentation.

---

## 📸 User Interface & Visual Design

The UI replicates the dark slate aesthetic of modern fintech and analytics platforms (such as Stripe and Nomad List):

- **Executive Color Palette:** Deep charcoal canvas (`#0c0d12`) in dark mode, crisp slate (`#f8fafc`) in light mode, elevated cards (`#181a26` / `#ffffff`), subtle borders (`rgba(255, 255, 255, 0.08)` / `#e2e8f0`), and high-contrast typography (`Inter`).
- **High-Contrast Light & Dark Modes:** All titles, metric values, subtexts, tables, dropdowns, and insight summaries adapt dynamically to theme selection so no text is obscured or washed out.
- **Professional Minimalist Glyphs:** Replaced casual emojis and box/calendar icons with sleek typographic symbols and indicators (`↗`, `ⓘ`, `›`, `•`, `✦`, `↑`, `↓`, `Δ`).
- **Interactive KPI Cards with Mini Sparklines:** Each primary KPI features a real-time delta pill badge (`+4.3%` green, `-12.9%` amber/red), bold primary metric (`$59,978.54`), prior-period comparison benchmark, and an embedded Plotly sparkline showing the recent trajectory alongside a faint baseline comparison curve.
- **Top Navigation Bar & Status Indicators:** Organization identifier (`Nomad Retail PROD v2.4`), live system heartbeat (`• Live System`), and clean navigation tabs (`Overview`, `Sales Analytics`, `Catalog Performance`, `Customer RFM`, `Executive Insights`, `Strategic Recommendations`, `Machine Learning`).

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Reactive executive interface and custom CSS engine |
| **Language** | Python 3.9+ | Core application and analytics logic |
| **Data Processing** | Pandas, NumPy | Cleaning, deduplication, cancellation handling, and aggregation |
| **Data Visualization** | Plotly Express & Graph Objects | Interactive charts, choropleth maps, 3D scatters, and sparklines |
| **Machine Learning** | Scikit-learn | RFM log-scaling, K-Means clustering, and 2D PCA projection |
| **Storage / Data** | CSV (UCI Online Retail standard) | File upload support and pre-bundled realistic multi-country dataset |

---

## 📂 Project Architecture

```
business/
├── app.py                      # Main Streamlit web application & UI layout
├── requirements.txt            # Project dependencies
├── README.md                   # Complete documentation and setup guide
├── generate_sample_data.py     # Deterministic generator for realistic retail data
├── data/
│   └── online_retail_sample.csv # 9,300+ sample transactions across 14 months
├── src/
│   ├── __init__.py
│   ├── ui_theme.py             # Custom CSS, dark/light themes, and Plotly styling
│   ├── data_loader.py          # Data ingestion, schema validation, cleaning, and date features
│   ├── analytics.py            # Business KPIs, period deltas, sparklines, sales & RFM logic
│   ├── ml_segmentation.py      # K-Means RFM clustering, PCA dimensionality reduction, and personas
│   └── insights_engine.py      # Automated executive observations & prescriptive playbooks
├── tests/
│   └── test_analytics.py       # Automated unit and integration test suite
└── .streamlit/
    ├── config.toml             # Streamlit server and dark theme configuration
    └── credentials.toml        # Local credentials bypass configuration
```

---

## 🚀 Quickstart & Installation Instructions

Follow these exact steps to run the dashboard on your machine:

### 1. Clone or Navigate to the Project Directory
```bash
cd /Users/apple/Desktop/business
```

### 2. Create and Activate a Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Generate or Refresh the Sample Dataset
A realistic dataset with ~9,300 transactions across 11 countries and 14 months is already generated. If you want to regenerate it:
```bash
python3 generate_sample_data.py
```

### 5. Run the Automated Unit Tests
Verify all data processing, KPI algorithms, and ML pipelines:
```bash
python3 tests/test_analytics.py
```

### 6. Launch the Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```
Or with custom port and host:
```bash
streamlit run app.py --server.port=8501
```

Once started, open your browser and navigate to:
```
http://localhost:8501
```

---

## 📊 Dashboard Modules & Features

### 🏠 A. Business Overview
- **Today's Trading Snapshot:** Gross checkout volume, yesterday comparison, peak transaction hour, settled balance ledger, and automated payout breakdown.
- **Real-Time KPI Cards with Sparklines:**
  - **Gross Volume ($):** Period net revenue with delta percentage pill badge and trendline.
  - **Total Orders:** Distinct paid invoices with period comparison.
  - **Active Customers:** Registered buyer count excluding guest checkouts.
  - **Average Order Value (AOV):** Mean basket size in USD.
  - **Best-Selling Product:** Highest revenue-generating SKU.
  - **Units Dispatched:** Total physical volume fulfilled.
  - **Return / Cancellation Rate:** Percentage and count of quarantined returns.
  - **Dispute Count:** Dispute and chargeback protection tracking (0.0%).
- **Velocity Timeline:** Interactive net sales line chart with a 7-day rolling smoothing curve.

### 📈 B. Sales Analytics
- **Monthly Revenue Performance:** Bar chart detailing month-by-month sales trajectory.
- **Month-over-Month (MoM) Growth:** Color-coded growth rate bars highlighting expansion vs. contraction cycles.
- **Geographic Distribution:** Interactive Plotly Choropleth world map and horizontal ranking of top export markets.
- **Intraday & Weekly Cadence:**
  - Sales velocity curve by hour of day (24h format).
  - Order volume distribution by day of the week.

### 📦 C. Product Analytics
- **Top 10 Sellers by Revenue:** Ranked bar chart of highest revenue contributors.
- **Top 10 Sellers by Volume:** Units fulfilled per product code.
- **Estimated Gross Profitability:** Product-level gross margins categorized by merchandise department.
- **Low-Performing / Tail SKUs:** Highlighting slow-moving products tying up working capital.
- **Category Donut Breakdown:** Revenue share across Home & Living, Kitchen & Dining, Gifts, Toys, Storage, etc.
- **Pareto 80/20 Analysis:** Cumulative catalog curve identifying the exact percentage of SKUs generating 80% of total revenue.

### 👥 D. Customer Analytics & RFM
- **New vs. Returning Customer Cohorts:** Stacked monthly revenue breakdown distinguishing repeat buyer loyalty from new acquisitions.
- **Purchase Frequency Distribution:** Histogram of orders per customer account.
- **RFM Segmentation Engine:**
  - **Recency (R):** Days elapsed since last order.
  - **Frequency (F):** Total number of distinct purchases.
  - **Monetary (M):** Total customer lifetime spend.
  - **Segment Assignment:** Champions, Loyal Customers, Potential Loyalists, New/Promising, At Risk, Can't Lose Them, and Hibernating.
- **VIP Customer Leaderboard:** Top 10 customer accounts ranked by spend, frequency, AOV, and days since last purchase.

### 💡 E. Automated Business Insights
Empirical, data-driven observations generated on the fly:
1. **Peak Revenue Month:** Highlights the highest-grossing period, percentage contribution, and seasonal correlation.
2. **Product Revenue Concentration:** Pinpoints top 5 SKU dependence and flagship product share.
3. **Customer Concentration (Whale Dependency):** Analyzes revenue generated by the top 5% of customer accounts.
4. **Market Dominance:** Identifies primary domestic vs. cross-border export opportunities.
5. **Declining Products Alert:** Detects velocity drop-offs in the last 45 days compared to prior periods.

### 🎯 F. Prescriptive Business Recommendations
Concrete, actionable strategies tailored to actual business metrics:
- **Supply Chain:** Safety stock buffers for high-velocity items to prevent out-of-stock lost revenue.
- **Merchandising:** Clearance bundles and promotions for bottom-performing inventory.
- **Customer Lifecycle:** 3-stage automated win-back workflows for "At Risk" customers and VIP early access drops for "Champions".
- **Seasonal Preparation:** Pre-holiday ramp-up schedules (Q3 procurement for Q4 surge).

### 🤖 5. Machine Learning (K-Means Clustering)
- **Feature Pipeline:** Log1p transformation to neutralize transactional right-skewness, followed by `StandardScaler` feature normalization.
- **Model:** Scikit-learn `KMeans` with configurable cluster counts ($k \in [2, 6]$).
- **Diagnostics:** Silhouette score evaluation and 2D Principal Component Analysis (PCA) variance tracking.
- **Visualizations:**
  - **3D Interactive RFM Scatter:** Rotate and inspect customer positions in Recency vs. Frequency vs. Monetary space.
  - **2D PCA Projection:** Clear cluster separation with component variance breakdowns.
- **Strategic Persona Profiles:** Dynamic cluster labeling into actionable cohorts:
  - *High-Value VIPs & Wholesalers*
  - *Loyal Regular Shoppers*
  - *New / Recent Shoppers*
  - *At-Risk / Lapsed Customers*
  - *Occasional Bargain Hunters*

---

## 🛡️ Data Ingestion, Cleaning & Validation

The application handles edge cases and data anomalies:
1. **Schema Mapping:** Intelligent column name matching (`InvoiceNo`, `Quantity`, `UnitPrice`, `InvoiceDate`, `CustomerID`, `Country`).
2. **Deduplication:** Automatic removal of exact duplicate transaction lines.
3. **Cancellation Isolation:** Separates credit notes (invoices starting with `C` or negative quantities) into a dedicated audit pool while keeping clean positive sales for core performance analysis.
4. **Missing Customer Handling:** Unassigned checkouts are assigned to `Guest Checkout` to preserve aggregate revenue while isolating individual RFM scoring to registered accounts.
5. **Dynamic Filters:** Instant reactivity for custom date spans, country multi-select, and category filtering.

---

## 🧪 Automated Testing

Run the included test suite to verify pipeline integrity:
```bash
source .venv/bin/activate
python3 tests/test_analytics.py
```
**Test Results:**
```
Ran 9 tests in 1.020s

OK
```

---

## 📄 License & Attribution

- Built with Streamlit, Plotly, Pandas, and Scikit-learn.
- Designed to replicate modern dark-mode fintech interfaces (Stripe / Nomad List).

