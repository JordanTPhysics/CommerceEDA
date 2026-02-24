import pandas as pd


def compute_rfm(df: pd.DataFrame, monetary_col: str | None = None) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) and segment per customer.
    Uses 'TotalCost' for Monetary if present, otherwise 'Revenue'.
    Returns DataFrame with Customer ID, Recency, Frequency, Monetary, R/F/M scores, RFM_score, Segment.
    """
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Remove missing customers and returns (optional but recommended)
    df = df.dropna(subset=["Customer ID"])
    df = df[df["Quantity"] > 0]

    monetary = monetary_col or ("TotalCost" if "TotalCost" in df.columns else "Revenue")
    if monetary not in df.columns:
        raise ValueError(f"DataFrame must have 'TotalCost' or 'Revenue' for Monetary; got {list(df.columns)}")

    # Create snapshot date (1 day after last transaction)
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    def recency_days(ser: pd.Series) -> int:
        return (snapshot_date - ser.max()).days

    # Aggregate to customer level (explicit dict avoids lambda/** agg issues)
    agg_dict = {
        "Recency": ("InvoiceDate", recency_days),
        "Frequency": ("Invoice", "nunique"),
        "Monetary": (monetary, "sum"),
    }
    rfm = df.groupby("Customer ID").agg(**agg_dict)

    # Create RFM scores (1–5)
    rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M_score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

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
    rfm = rfm.sort_values(by="Monetary", ascending=False)
    return rfm


def main(df: pd.DataFrame) -> pd.DataFrame:
    """Run RFM segmentation and return the result (for standalone use)."""
    return compute_rfm(df)