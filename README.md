# CommerceEDA

Exploratory data analysis (EDA) and customer segmentation for e-commerce transaction data. This project provides an interactive Streamlit dashboard to explore orders, revenue, product trends, and RFM-based customer segments—including handling of credit/return invoices.

## Dataset

The app uses the [Supermarket / E-commerce dataset from Kaggle](https://www.kaggle.com/datasets/saurabhbadole/supermarket-data). Place the main transaction file at:

```
data/supermarket_data.csv
```

Expected columns: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`. Credit/return invoices are identified by an `Invoice` ID starting with `C` (e.g. `C536391`).

## Setup

**Requirements:** Python 3.10+ (for `str | None` type hints in `customer_rfm.py`).

1. Clone the repo and create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the app

From the project root:

```bash
streamlit run ecommerce_explore.py
```

The dashboard opens in your browser (default: http://localhost:8501).

## Features

- **Summary & geography** — Total orders, revenue, countries, and average order value; credits overview (invoices credited, amount credited); orders and revenue by country.
- **Products** — Top 10 items by quantity per month/year (with Prev/Next), item catalog (total sold, average price).
- **Trends** — Order frequency and revenue by month, weekday, and hour (Plotly charts).
- **Customer analysis (RFM)** — Recency, frequency, and monetary segmentation with net revenue (credits subtracted). Segment breakdown, pie chart, and 3D scatter (linear/log scale). Segment definitions (Champions, Loyal, New, At Risk, Lost, Others).
- **Debug: RFM dataframe** — Filterable view of the RFM table (by segment, Customer ID, recency, net revenue).
- **Customer search** — Enter a Customer ID to see total value (net), invoice count, credits count, amount credited, and a table of **all invoices** for that customer (date, revenue, country, line count).

## Project structure

```
CommerceEDA/
├── data/
│   └── supermarket_data.csv   # Place Kaggle dataset here
├── ecommerce_explore.py       # Streamlit dashboard
├── customer_rfm.py            # RFM computation (with credit handling)
├── requirements.txt
└── README.md
```

## RFM and credits

- **Recency** — Days since last *purchase* (credit invoices excluded).
- **Frequency** — Purchase invoices per month; credit invoices are not counted. If a customer’s activity spans only one month, frequency is their invoice count for that month.
- **Monetary** — Net revenue (sales minus credits) per purchase invoice.

Only customers with non-empty `Customer ID` and total net revenue ≥ 0 are included in the invoice-level data used for RFM in the app.
