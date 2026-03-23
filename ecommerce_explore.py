import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from html import escape

from streamlit import config

from customer_rfm import compute_rfm, _is_credit_invoice

st.set_page_config(page_title="E-commerce Dashboard", layout="wide")

# --- Analysis text (edit these to add your comments) ---
ANALYSIS_COUNTRY = "Clearly this is a UK based company as the dataset is heavily skewed towards UK sales. The summary page does show a few other high scorers like Australia and Lebanon which have substantial order value, and Netherlands, France, Germany and Ireland generating most of the revenue not going to UK. We could add a country filter to see if the trends in this dataset vary by country later."
ANALYSIS_TOP_ITEMS = "Popular items change regularly with a few repeat winners, it also shows a pretty even distribution of individual consumers and items sold wholesale, even though the bulk items generally come on top"
ANALYSIS_TRENDS_MONTHLY = "Monthly trends show a clear rise in orders and revenue over the course of a year, peaking in November just before the holiday season, there's also some interesting activity as the UK tax year comes to an end, suggesting procurement teams clearing their budget surplus"
ANALYSIS_TRENDS_WEEKLY = "Not much stands out on the weekly chart, but it seems clear the shop isn't open on Saturday, but is open on Sunday, the least profitable day of the week. It probably makes more sense to switch these 2 days to avoid losing revenue to competition"
ANALYSIS_TRENDS_HOURLY = "We've seen hints at the large contribution of wholesale orders to the total revenue but this chart makes it clear with jumps in revenue at 10am and 3pm while orders remain steady"
ANALYSIS_SEGMENTATION = "Champions tend to be the wholesale orders, they are most loyal, spend the most money and make regular purchases, of all groups they contribute most to revenue while being only 2% of customers Consider giving discounts on wholesale orders to reward these loyal customers. Over half of the customers are new or unidentified and also contribute a lot of the revenue, it would be interesting to see the products they buy and create offers that make them loyal customers. At risk and lost customers are about 25% of the customer base but contribute less to revenue than any other group even when combined."

def analysis_span(text: str, title: str = "Analysis") -> None:
    if not text.strip():
        return
    safe = escape(text.strip()).replace("\n", "<br>")
    st.markdown(
        f'<div style="background:#18191a; color:#ffffff; border-left:4px solid #1f77b4; padding:0.75rem 1rem; margin:0.5rem 0; border-radius:0 4px 4px 0;"><small style="color:#fafdfe; font-weight:600;">{title}</small><br>{safe}</div>',
        unsafe_allow_html=True,
    )


st.title("E-commerce Data Exploration")

st.divider()


analysis_span("Today I will showcase how much customer information can be gleaned from a dataset of transactions for an online retailer. Without knowing any personally identifiable information or demographic data, we will gain insight into the customers' spending habits, what they like and dislike, and growth opportunities for the store. " ,"Exploratory Data Analysis of E-commerce Data from Kaggle")

link1, link2, space = st.columns([1,1,4])
with link1:
    st.link_button("Dataset", "https://www.kaggle.com/datasets/saurabhbadole/supermarket-data")
with link2:
    st.link_button("GitHub", "https://github.com/JordanTPhysics/CommerceEDA")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/supermarket_data.csv")

    df = df[df["Country"] != "Israel"] #remove the trash

    df["Revenue"] = df["Quantity"] * df["Price"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%d/%m/%y %H:%M")
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Weekday"] = df["InvoiceDate"].dt.weekday
    df["Hour"] = df["InvoiceDate"].dt.hour
    return df


@st.cache_data
def get_top_items_per_period():
    """Precompute top 10 items by quantity for every (Year, Month). Cached so Prev/Next only does a lookup."""
    df = load_data()
    periods = df[["Year", "Month"]].drop_duplicates().sort_values(["Year", "Month"])
    rows = []
    for _, row in periods.iterrows():
        y, m = int(row["Year"]), int(row["Month"])
        month_df = df[(df["Year"] == y) & (df["Month"] == m)]
        top = (
            month_df.groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top.columns = ["Item", "Quantity sold"]
        top["Year"], top["Month"] = y, m
        top["Rank"] = range(1, len(top) + 1)
        rows.append(top)
    if not rows:
        return pd.DataFrame(columns=["Year", "Month", "Rank", "Item", "Quantity sold"])
    return pd.concat(rows, ignore_index=True)[["Year", "Month", "Rank", "Item", "Quantity sold"]]


@st.cache_data
def get_rfm(df: pd.DataFrame):
    """Cached RFM segmentation from main dataset (uses Revenue for Monetary)."""
    return compute_rfm(df, monetary_col="Revenue")


df = load_data()

# Only include customers whose total revenue (across all invoices) is >= 0
customer_total_revenue = df.groupby("Customer ID")["Revenue"].sum()
customers_non_negative = customer_total_revenue[customer_total_revenue >= 0].index
invoice_df = df[(df["Customer ID"] != "") & (df["Customer ID"].isin(customers_non_negative))]
invoice_agg = {
    "Invoice": ("Invoice", "first"),
    "Revenue": ("Revenue", "sum"),
    "Customer ID": ("Customer ID", "first"),
    "InvoiceDate": ("InvoiceDate", "first"),
}
invoice_df = invoice_df.groupby("Invoice").agg(**invoice_agg)

periods = (
    df[["Year", "Month"]].drop_duplicates().sort_values(["Year", "Month"]).astype(int)
)
period_list = periods.values.tolist()
month_names = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
default_period_idx = len(period_list) - 1 if period_list else None

by_country = (df.groupby("Country", as_index=False)
    .agg(Orders=("Invoice", "nunique"), Revenue=("Revenue", "sum"), Customers=("Customer ID", "nunique"))
    .sort_values("Revenue", ascending=False)
)
by_country["Average Order Value"] = by_country["Revenue"] / by_country["Orders"]

by_month = (
    df.groupby(["Month"], as_index=False)
    .agg(Orders=("Invoice", "nunique"), Revenue=("Revenue", "sum"), Customers=("Customer ID", "nunique"))
)
by_month["MonthName"] = by_month["Month"].apply(lambda x: month_names[x - 1])

# Weekday: ensure all 7 days appear (fill missing with 0)
by_weekday = df.groupby(["Weekday"], as_index=False).agg(Orders=("Invoice", "nunique"), Revenue=("Revenue", "sum"))
full_weekdays = pd.DataFrame({"Weekday": range(7)})
by_weekday = full_weekdays.merge(by_weekday, on="Weekday", how="left").fillna(0)
by_weekday["Orders"] = by_weekday["Orders"].astype(int)
by_weekday["WeekdayName"] = by_weekday["Weekday"].apply(lambda x: weekday_names[x])

# Hour: ensure all 24 hours appear (fill missing with 0)
by_hour = df.groupby(["Hour"], as_index=False).agg(Orders=("Invoice", "nunique"), Revenue=("Revenue", "sum"))
full_hours = pd.DataFrame({"Hour": range(24)})
by_hour = full_hours.merge(by_hour, on="Hour", how="left").fillna(0)
by_hour["Orders"] = by_hour["Orders"].astype(int)
by_hour["HourLabel"] = by_hour["Hour"].apply(lambda x: f"{x:02d}:00")

by_item = (
    df.groupby(["StockCode", "Description"], as_index=False)
        .agg(TotalSold=("Quantity", "sum"), AveragePrice=("Price", "mean"))
        .sort_values("TotalSold", ascending=False)
) 
# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Total orders", f"{df['Invoice'].nunique():,}")
with col2: st.metric("Total revenue", f"£{df['Revenue'].sum():,.0f}")
with col3: st.metric("Countries", by_country["Country"].nunique())
with col4: st.metric("Average Order Value", f"£{by_country['Average Order Value'].mean():,.0f}")

# Credit invoices overview
is_credit = _is_credit_invoice(df["Invoice"])
credit_df = df[is_credit]
total_invoices_credited = credit_df["Invoice"].nunique()
total_amount_credited = credit_df["Revenue"].sum()  # negative; display as positive "amount credited"
st.subheader("Credits overview")
cred_c1, cred_c2 = st.columns(2)
with cred_c1: st.metric("Total invoices credited", f"{total_invoices_credited:,}")
with cred_c2: st.metric("Total amount credited", f"£{abs(total_amount_credited):,.0f}")

st.header("Orders and revenue by country")
st.dataframe(
    by_country.style.format({"Revenue": "£{:,.0f}", "Orders": "{:,.0f}", "Average Order Value": "£{:,.0f}"}, thousands=","),
    width='stretch',
    hide_index=True,
)
analysis_span(ANALYSIS_COUNTRY)

st.divider()
item_quantities, item_catalog = st.columns([1, 1])

with item_quantities:
    st.subheader("Top 10 most popular items by month and year")
    if default_period_idx is not None:
        if "period_idx" not in st.session_state:
            st.session_state["period_idx"] = default_period_idx
        idx = st.session_state["period_idx"]
        year, month = period_list[idx]
        label = f"{month_names[month - 1]} {year}"
        can_go_prev = idx > 0
        can_go_next = idx < len(period_list) - 1

        top_items_all = get_top_items_per_period()
        top_items = top_items_all[
            (top_items_all["Year"] == year) & (top_items_all["Month"] == month)
        ][["Rank", "Item", "Quantity sold"]].copy()

        fig_top = px.bar(
            top_items,
            x="Quantity sold",
            y="Item",
            orientation="h",
            title=f"Top 10 items by quantity sold — {label}",
            labels={"Quantity sold": "Units sold"},
        )
        fig_top.update_layout(
            height=400,
            font=dict(size=16),
            title_font=dict(size=20),
            xaxis=dict(
                domain=[0.5, 1],
                tickfont=dict(size=12),
                title_font=dict(size=18),
            ),
            yaxis=dict(
                categoryorder="total ascending",
                tickfont=dict(size=12),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            autosize=True,
        )
        st.plotly_chart(fig_top)

        col_prev, col_next, col_period = st.columns([2, 2, 3])
        with col_prev:
            if st.button("◀ Previous", disabled=not can_go_prev, width='stretch') and can_go_prev:
                st.session_state["period_idx"] = idx - 1
                st.rerun()
        with col_period: st.markdown(f"**{label}**")
        with col_next:
            if st.button("Next ▶", disabled=not can_go_next, width='stretch') and can_go_next:
                st.session_state["period_idx"] = idx + 1
                st.rerun()


    else:
        st.info("No date data available to filter by month and year.")

with item_catalog:
    st.subheader("Item Catalog")
    st.dataframe(
        by_item.style.format({"TotalSold": "{:,.0f}", "AveragePrice": "£{:,.2f}"}, thousands=","),
        width='stretch',
        hide_index=True,
    )

analysis_span(ANALYSIS_TOP_ITEMS)

st.divider()

st.header("Order and Revenue Trends")

fig_orders_monthly = go.Figure()
fig_orders_monthly.add_trace(
    go.Scatter(x=by_month["MonthName"], y=by_month["Orders"], name="Orders", line=dict(color="#1f77b4"))
)
fig_orders_monthly.add_trace(
    go.Scatter(x=by_month["MonthName"], y=by_month["Revenue"], name="Revenue", yaxis="y2", line=dict(color="#ff7f0e"))
)
fig_orders_monthly.update_layout(
    title="Order Frequency and Revenue - Yearly",
    yaxis=dict(title="Orders", side="left"),
    yaxis2=dict(title="Revenue (£)", side="right", overlaying="y", tickformat=",.0f"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400,
)
st.plotly_chart(fig_orders_monthly, width='stretch')
analysis_span(ANALYSIS_TRENDS_MONTHLY)

fig_orders_weekdays = go.Figure()
fig_orders_weekdays.add_trace(
    go.Scatter(x=by_weekday["WeekdayName"], y=by_weekday["Orders"], name="Orders", line=dict(color="#1f77b4"))
)
fig_orders_weekdays.add_trace(
    go.Scatter(x=by_weekday["WeekdayName"], y=by_weekday["Revenue"], name="Revenue", yaxis="y2", line=dict(color="#ff7f0e"))
)
fig_orders_weekdays.update_layout(
    title="Order Frequency and Revenue - Weekly",
    xaxis=dict(
        categoryorder="array",
        categoryarray=by_weekday["WeekdayName"].tolist(),
        tickmode="array",
        tickvals=by_weekday["WeekdayName"].tolist(),
    ),
    yaxis=dict(title="Orders", side="left"),
    yaxis2=dict(title="Revenue (£)", side="right", overlaying="y", tickformat=",.0f"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400,
)
st.plotly_chart(fig_orders_weekdays, width='stretch')
analysis_span(ANALYSIS_TRENDS_WEEKLY)


fig_orders_hourly = go.Figure()
fig_orders_hourly.add_trace(
    go.Scatter(x=by_hour["HourLabel"], y=by_hour["Orders"], name="Orders", line=dict(color="#1f77b4"))
)
fig_orders_hourly.add_trace(
    go.Scatter(x=by_hour["HourLabel"], y=by_hour["Revenue"], name="Revenue", yaxis="y2", line=dict(color="#ff7f0e"))
)
fig_orders_hourly.update_layout(
    title="Order Frequency and Revenue - Daily",
    xaxis=dict(
        categoryorder="array",
        categoryarray=by_hour["HourLabel"].tolist(),
        tickmode="array",
        tickvals=by_hour["HourLabel"].tolist(),
    ),
    yaxis=dict(title="Orders", side="left"),
    yaxis2=dict(title="Revenue (£)", side="right", overlaying="y", tickformat=",.0f"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400,
)
st.plotly_chart(fig_orders_hourly, width='stretch')
analysis_span(ANALYSIS_TRENDS_HOURLY)

# --- Customer segmentation (RFM) ---
st.divider()
st.header("Customer Analysis")


rfm = get_rfm(invoice_df)

seg_counts = rfm.groupby("Segment", as_index=False).agg(
    Customers=("Customer ID", "count"),
    Total_Monetary=("NetRevenue", "sum"),
    Invoice_Cost_Avg=("NetRevenue", "mean"),
    Avg_Recency=("Recency", "mean"),
    Monthly_Invoices_Avg=("InvoicesPerMonth", "mean"),
).round(1)
seg_counts = seg_counts.sort_values("Customers", ascending=False)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total customers", f"{len(rfm):,}")
with c2:
    st.metric("Segments", rfm["Segment"].nunique())
with c3:
    st.metric("Champions (555)", f"{len(rfm[rfm['Segment'] == 'Champions']):,}")
with c4:
    st.metric("Total segment value (£)", f"{rfm['NetRevenue'].sum():,.0f}")

st.subheader("Overview by segment")
st.dataframe(
    seg_counts.style.format(
        {"Customers": "{:,.0f}", "Total_Monetary": "£{:,.0f}", "Invoice_Cost_Avg": "£{:,.0f}", "Avg_Recency": "{:.1f}", "Monthly_Invoices_Avg": "{:.1f}"},
        thousands=",",
    ),
    width='stretch',
    hide_index=True,
)

seg_pie_col, seg_3d_col = st.columns([1, 1])
with seg_pie_col:
    fig_seg_pie = px.pie(
        seg_counts,
        values="Customers",
        names="Segment",
        title="Share of customers by segment",
        hole=0.2,
    )
    fig_seg_pie.update_layout(height=420)
    st.plotly_chart(
        fig_seg_pie,
        width='stretch',
        config={"color_discrete_sequence": px.colors.qualitative.Set2}
    )

if "plot_type" not in st.session_state:
    st.session_state["plot_type"] = "linear"

def toggle_log_scale():
    st.session_state["plot_type"] = "log" if st.session_state["plot_type"] == "linear" else "linear"

with seg_3d_col:
    rfm_log = rfm.assign(
        log_Recency=np.log1p(rfm["Recency"]),
        log_Frequency=np.log1p(rfm["InvoicesPerMonth"]),
        log_Monetary=np.log1p(rfm["InvoiceAverage"]),
    )
    chart_df = rfm_log if st.session_state["plot_type"] == "log" else rfm
    fig_rfm_3d = px.scatter_3d(
        chart_df,
        x="log_Recency" if st.session_state["plot_type"] == "log" else "Recency",
        y="log_Frequency" if st.session_state["plot_type"] == "log" else "InvoicesPerMonth",
        z="log_Monetary" if st.session_state["plot_type"] == "log" else "InvoiceAverage",
        color="Segment",
        title=f"RFM space by segment {'(log scale)' if st.session_state['plot_type'] == 'log' else ''}",
        labels={
            "Recency": "Recency",
            "InvoicesPerMonth": "Invoices per month",
            "InvoiceAverage": "Average invoice value",
            "log_Recency": "log(1 + Recency)",
            "log_Frequency": "log(1 + InvoicesPerMonth)",
            "log_Monetary": "log(1 + InvoiceAverage)",
        },
        hover_data={
            "Customer ID": True,
            "Segment": True,
            "R_score": True,
            "F_score": True,
            "M_score": True,
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
        opacity=0.7,
    )
    fig_rfm_3d.update_traces(marker=dict(size=3))
    fig_rfm_3d.update_layout(height=420, margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig_rfm_3d, width='stretch')
    st.button("Log scale" if st.session_state["plot_type"] == "linear" else "Linear scale", on_click=toggle_log_scale)

st.subheader("Segment definitions")
st.markdown(
    "- **Champions** (555): Best customers — recent, frequent, high spend. Reward and retain.\n"
    "- **Loyal Customers**: Recent and frequent; nurture with loyalty programs.\n"
    "- **New Customers**: Recent but low frequency; onboard and cross-sell.\n"
    "- **At Risk**: Used to order often but not recently; win-back campaigns.\n"
    "- **Lost**: Low recency and frequency; consider low-cost reactivation or deprioritise.\n"
    "- **Others**: Middle ground; encourage with targeted offers."
)
analysis_span(ANALYSIS_SEGMENTATION)

st.divider()
st.header("Exploratory Data Analysis Summary")

analysis_span("This data app gives a surface level analysis of the strengths and weaknesses of this Wholesaler. There is potential to look more deeply into the data and customer trends to increase sales and revenue. If you're interested you can fork the GitHub and try some yourself:")
st.markdown(
   "1. Analyse the performance and prices of individual popular products over time to see when discounts are most effective\n"
   "2. Devise a way to categorise products into groups based on being sold together or by the same customers, then seeing how product groups perform.\n"
   "3. Add filtering to see how the data varies by country or product category, showing which countries are more successful and why.",
   text_alignment="center",
   width="content",
)

with st.expander("Debug: Customer Search", expanded=True):
    rfm_view = rfm.copy()
    seg_options = sorted(rfm["Segment"].unique().tolist())
    segments_filter = st.multiselect(
        "Segment",
        options=seg_options,
        default=seg_options,
        key="rfm_debug_segment",
    )
    rfm_view = rfm_view[rfm_view["Segment"].isin(segments_filter)]
    debug_cid = st.text_input("Customer ID (optional)", placeholder="e.g. 17850", key="rfm_debug_cid")
    if debug_cid and debug_cid.strip():
        try:
            cid_filter = int(debug_cid.strip())
            rfm_view = rfm_view[rfm_view["Customer ID"] == cid_filter]
        except ValueError:
            st.caption("Customer ID must be a number; showing all segments above.")
    col_rec, col_val = st.columns(2)
    with col_rec:
        rec_min, rec_max = int(rfm["Recency"].min()), int(rfm["Recency"].max())
        rec_range = st.slider("Recency (days)", rec_min, rec_max, (rec_min, rec_max), key="rfm_debug_recency")
        rfm_view = rfm_view[(rfm_view["Recency"] >= rec_range[0]) & (rfm_view["Recency"] <= rec_range[1])]
    with col_val:
        val_min, val_max = int(rfm["NetRevenue"].min()), int(rfm["NetRevenue"].max())
        val_range = st.slider("NetRevenue (£)", val_min, val_max, (val_min, val_max), key="rfm_debug_value")
        rfm_view = rfm_view[(rfm_view["NetRevenue"] >= val_range[0]) & (rfm_view["NetRevenue"] <= val_range[1])]
    st.caption(f"Showing {len(rfm_view):,} of {len(rfm):,} customers.")
    st.dataframe(
        rfm_view.style.format(
            {"Customer ID": "{:,.0f}", "NetRevenue": "£{:,.0f}", "InvoiceAverage": "£{:,.2f}", "MonthsActive": "{:.2f}", "InvoicesPerMonth": "{:.2f}", "Recency": "{:.0f}"},
            thousands=",",
        ),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Customer search")
customer_id_input = st.text_input("Customer ID", placeholder="e.g. 17850", key="customer_search_id")
if customer_id_input and customer_id_input.strip():
    try:
        search_cid = int(customer_id_input.strip())
    except ValueError:
        st.warning("Please enter a numeric Customer ID.")
    else:
        cust_df = df[df["Customer ID"] == search_cid]
        if cust_df.empty:
            st.info(f"No data found for Customer ID **{search_cid}**.")
        else:
            cust_is_credit = _is_credit_invoice(cust_df["Invoice"])
            cust_sales = cust_df[~cust_is_credit]
            cust_credits = cust_df[cust_is_credit]
            net_value = cust_df["Revenue"].sum()
            num_purchases = cust_sales["Invoice"].nunique()
            num_credits = cust_credits["Invoice"].nunique()
            amount_credited = cust_credits["Revenue"].sum()
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Total value (net)", f"£{net_value:,.0f}")
            with m2: st.metric("Invoices", f"{num_purchases:,}")
            with m3: st.metric("Credits", f"{num_credits:,}")
            with m4: st.metric("Total credited", f"£{abs(amount_credited):,.0f}")

            st.subheader("Invoices for this customer")
            inv_agg = {
                "InvoiceDate": ("InvoiceDate", "first"),
                "Revenue": ("Revenue", "sum"),
                "Country": ("Country", "first"),
                "Unique Items": ("StockCode", "count"),
            }
            cust_invoices = cust_df.groupby("Invoice").agg(**inv_agg).sort_values("InvoiceDate")
            cust_invoices = cust_invoices.reset_index()
            st.dataframe(
                cust_invoices.style.format(
                    {"Revenue": "£{:,.2f}", "Unique Items": "{:,.0f}"},
                    thousands=",",
                ),
                use_container_width=True,
                hide_index=True,
            )

