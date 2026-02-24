import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from html import escape

from customer_rfm import compute_rfm

st.set_page_config(page_title="E-commerce Dashboard", layout="wide")

# --- Analysis text (edit these to add your comments) ---
ANALYSIS_COUNTRY = "Clearly this is a UK based company as the dataset is heavily skewed towards UK sales. The summary page does show a few other high scorers like Australia and Lebanon which have substantial order value, and Netherlands, France, Germany and Ireland generating most of the revenue not going to UK. We could add a country filter to see if the trends in this dataset vary by country later."
ANALYSIS_TOP_ITEMS = "Popular items change regularly with a few repeat winners, it also shows a pretty even distribution of individual consumers and items sold wholesale, even though the bulk items generally come on top"
ANALYSIS_TRENDS_MONTHLY = "Monthly trends show a clear rise in orders and revenue over the course of a year, peaking in November just before the holiday season, there's also some interesting activity as the UK tax year comes to an end, suggesting procurement teams clearing their budget surplus"
ANALYSIS_TRENDS_WEEKLY = "Not much stands out on the weekly chart, but it seems clear the shop isn't open on Saturday, but is open on Sunday, the least profitable day of the week. It probably makes more sense to switch these 2 days to avoid losing revenue to competition"
ANALYSIS_TRENDS_HOURLY = "We've seen hints at the large contribution of wholesale orders to the total revenue but this chart makes it clear with jumps in revenue at 10am and 3pm while orders remain steady"


def analysis_span(text: str) -> None:
    if not text.strip():
        return
    safe = escape(text.strip()).replace("\n", "<br>")
    st.markdown(
        f'<div style="background:#f8f9fa; color:#111111; border-left:4px solid #1f77b4; padding:0.75rem 1rem; margin:0.5rem 0; border-radius:0 4px 4px 0;"><small style="color:#111111; font-weight:600;">Analysis</small><br>{safe}</div>',
        unsafe_allow_html=True,
    )


st.title("E-commerce Data Exploration")

st.divider()

head1, head2 = st.columns([4, 1])
with head1: st.write("Exploratory Data Analysis of E-commerce Data from Kaggle, by Jordan"),
with head2: st.link_button("Dataset", "https://www.kaggle.com/datasets/saurabhbadole/supermarket-data")
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
def get_rfm():
    """Cached RFM segmentation from main dataset (uses Revenue for Monetary)."""
    df = load_data()
    return compute_rfm(df, monetary_col="Revenue")


df = load_data()
#st.sidebar.title("Filters")
#country_filter = st.sidebar.multiselect("Select Country", df["Country"].unique())

#df = df[df["Country"].isin(country_filter)] if country_filter else df

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
    .agg(Orders=("Invoice", "nunique"), Revenue=("Revenue", "sum"))
    .sort_values("Revenue", ascending=False)
)
by_country["Average Order Value"] = by_country["Revenue"] / by_country["Orders"]

by_month = (
    df.groupby(["Month"], as_index=False)
    .agg(Orders=("Invoice", "nunique"), Revenue=("Revenue", "sum"))
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
st.header("Customer segmentation")
ANALYSIS_SEGMENTATION = ""

rfm = get_rfm()

# Overview metrics
seg_counts = rfm.groupby("Segment", as_index=False).agg(
    Customers=("Customer ID", "count"),
    Total_Monetary=("Monetary", "sum"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
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
    st.metric("Total segment value (£)", f"{rfm['Monetary'].sum():,.0f}")

st.subheader("Overview by segment")
st.dataframe(
    seg_counts.style.format(
        {"Customers": "{:,.0f}", "Total_Monetary": "£{:,.0f}", "Avg_Recency": "{:.1f}", "Avg_Frequency": "{:.1f}"},
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
    st.plotly_chart(fig_seg_pie, width='stretch')

with seg_3d_col:
    fig_rfm_3d = px.scatter_3d(
        rfm,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Segment",
        title="RFM space by segment",
        labels={"Recency": "Recency (days)", "Frequency": "Frequency", "Monetary": "Monetary (£)"},
        color_discrete_sequence=px.colors.qualitative.Set2,
        opacity=0.7,
    )
    fig_rfm_3d.update_traces(marker=dict(size=3))
    fig_rfm_3d.update_layout(height=420, margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig_rfm_3d, width='stretch')

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
