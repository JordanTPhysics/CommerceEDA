import pandas as pd


def _is_credit_invoice(invoice_series: pd.Series) -> pd.Series:
    """Return True for rows that are credit/return invoices (e.g. Invoice starting with 'C')."""
    return invoice_series.astype(str).str.strip().str.upper().str.startswith("C")


def compute_rfm(df: pd.DataFrame, monetary_col: str | None = None) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) and segment per customer.
    - Recency: days since last purchase (excludes credit-only invoices).
    - Frequency: purchase invoices per month (credit invoices not counted as purchases).
    - Monetary: net value per purchase invoice (total revenue minus credits, divided by purchase invoice count).
    Uses 'TotalCost' or 'Revenue' for value. Credit invoices (e.g. Invoice starting with 'C') reduce monetary value.
    Returns DataFrame with Customer ID, Recency, Frequency, Monetary, R/F/M scores, RFM_score, Segment.
    """
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["IsCredit"] = _is_credit_invoice(df["Invoice"])

    # Remove missing customers
    df = df.dropna(subset=["Customer ID"])

    # Sales only: exclude credit invoices (used for recency, frequency, and "purchase" count)
    sales = df[~df["IsCredit"]].copy()
    # For monetary we use full df so credits reduce net value

    monetary = monetary_col or ("TotalCost" if "TotalCost" in df.columns else "Revenue")
    if monetary not in df.columns:
        raise ValueError(f"DataFrame must have 'TotalCost' or 'Revenue' for Monetary; got {list(df.columns)}")

    # Create snapshot date (1 day after last transaction)
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    days_per_month = 30.44

    def recency_days(ser: pd.Series) -> int:
        return (snapshot_date - ser.max()).days

    def first_date(ser: pd.Series) -> pd.Timestamp:
        return ser.min()
    
    def last_date(ser: pd.Series) -> pd.Timestamp:
        return ser.max()

    # Recency and invoice count from sales only; net value from all rows (credits reduce it)
    rfm_sales = sales.groupby("Customer ID").agg(
        Recency=("InvoiceDate", recency_days),
        InvoiceCount=("Invoice", "nunique"),
        FirstDate=("InvoiceDate", first_date),
        LastDate=("InvoiceDate", last_date),
    )
    rfm_value = df.groupby("Customer ID").agg(NetRevenue=(monetary, "sum"))
    rfm = rfm_sales.join(rfm_value)

    # Frequency = invoices per month; when activity spans only 1 month, use invoice count as-is
    span_days = (rfm["LastDate"] - rfm["FirstDate"]).dt.days
    span_months = span_days / days_per_month
    rfm["MonthsActive"] = span_months.clip(lower=1)  # at most 1 month → treat as 1 month so InvoicesPerMonth = invoice count
    rfm["InvoicesPerMonth"] = rfm["InvoiceCount"] / rfm["MonthsActive"]
    rfm["InvoiceAverage"] = rfm["NetRevenue"] / rfm["InvoiceCount"]
    rfm = rfm.drop(columns=["InvoiceCount", "FirstDate", "LastDate", "MonthsActive"])

    # Create RFM scores (1–5)
    rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm['F_score'] = pd.qcut(rfm['InvoicesPerMonth'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M_score'] = pd.qcut(rfm['InvoiceAverage'], 5, labels=[1,2,3,4,5])

    # Combine scores
    rfm['RFM_score'] = (
        rfm['R_score'].astype(str) +
        rfm['F_score'].astype(str) +
        rfm['M_score'].astype(str)
    )

    # Basic segmentation
    def segment(row):
        if row['RFM_score'] == '555':
            return 'Champions'
        elif row['R_score'] >= 4 and row['F_score'] >= 4:
            return 'Loyal Customers'
        elif row['R_score'] >= 4 and row['F_score'] <= 2:
            return 'New Customers'
        elif row['R_score'] <= 2 and row['F_score'] >= 4:
            return 'At Risk'
        elif row['R_score'] <= 2 and row['F_score'] <= 2:
            return 'Lost'
        else:
            return 'Others'

    rfm['Segment'] = rfm.apply(segment, axis=1)

    # Reset index for easier viewing
    rfm = rfm.reset_index()

    # Sort by value
    rfm = rfm.sort_values(by="NetRevenue", ascending=False)
    return rfm


def main(df: pd.DataFrame) -> pd.DataFrame:
    """Run RFM segmentation and return the result (for standalone use)."""
    return compute_rfm(df)