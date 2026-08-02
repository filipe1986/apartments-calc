# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import statsmodels

st.set_page_config(
    page_title="Apartments Portfolio Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/filipe1986/apartments-calc/main/apartments.csv"
    df = pd.read_csv(url)

    # Clean & enrich
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["sqft"] = pd.to_numeric(df["sqft"], errors="coerce")
    df["bed"] = pd.to_numeric(df["bed"], errors="coerce")
    df["bath"] = pd.to_numeric(df["bath"], errors="coerce")
    df["y_built"] = pd.to_numeric(df["y_built"], errors="coerce")
    df["ren_y"] = pd.to_numeric(df["ren_y"], errors="coerce")

    df["price_per_sqft"] = df["price"] / df["sqft"]
    df["age"] = 2026 - df["y_built"]                       # current year from data context
    df["renovated"] = df["ren_y"].notna()
    df["years_since_reno"] = 2026 - df["ren_y"]
    df["fire"] = df["fire"].map({"t": True, "f": False})
    df["size_band"] = pd.cut(
        df["sqft"],
        bins=[0, 1000, 1500, 2000, 2500, 5000],
        labels=["<1k", "1-1.5k", "1.5-2k", "2-2.5k", "2.5k+"],
    )
    df["created_at"] = pd.to_datetime(df["created_at"])

    return df

df = load_data()


st.sidebar.header("Filters")

status_opts = sorted(df["listing_status"].unique())
selected_status = st.sidebar.multiselect("Listing Status", status_opts, default=status_opts)

neigh_opts = sorted(df["neighborhood"].unique())
selected_neigh = st.sidebar.multiselect("Neighborhood", neigh_opts, default=neigh_opts)

bed_opts = sorted(df["bed"].dropna().unique())
selected_beds = st.sidebar.multiselect("Bedrooms", bed_opts, default=bed_opts)

price_min, price_max = float(df["price"].min()), float(df["price"].max())
price_range = st.sidebar.slider("Price range ($)", price_min, price_max, (price_min, price_max))

mask = (
    df["listing_status"].isin(selected_status)
    & df["neighborhood"].isin(selected_neigh)
    & df["bed"].isin(selected_beds)
    & df["price"].between(price_range[0], price_range[1])
)
filtered = df[mask].copy()


st.title("Apartments Portfolio Dashboard")
st.caption("Source: filipe1986/apartments-calc · Focus on inventory, pricing power & segment performance")

# KPI calculations
n_listings = len(filtered)
n_active = (filtered["listing_status"] == "active").sum()
n_pending = (filtered["listing_status"] == "pending").sum()
n_sold = (filtered["listing_status"] == "sold").sum()
avg_price = filtered["price"].mean()
median_price = filtered["price"].median()
avg_ppsf = filtered["price_per_sqft"].mean()
pct_pending_or_sold = (n_pending + n_sold) / n_listings * 100 if n_listings else 0

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Listings", f"{n_listings:,}")
k2.metric("Active", f"{n_active:,}")
k3.metric("Pending", f"{n_pending:,}")
k4.metric("Sold", f"{n_sold:,}")
k5.metric("Avg Price", f"${avg_price:,.0f}")
k6.metric("Avg $/sqft", f"${avg_ppsf:,.0f}")

st.markdown(f"**Liquidity signal**: {pct_pending_or_sold:.1f}% of filtered inventory is pending or sold")


tab1, tab2, tab3, tab4 = st.tabs([
    "Inventory Mix",
    "Pricing Analysis",
    "Neighborhood Performance",
    "Feature Premiums",
])

with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        status_counts = filtered["listing_status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig = px.pie(status_counts, names="status", values="count", title="Status Mix", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        bed_status = (
            filtered.groupby(["bed", "listing_status"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            bed_status,
            x="bed",
            y="count",
            color="listing_status",
            barmode="group",
            title="Listings by Bedrooms & Status",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.histogram(
            filtered,
            x="price",
            nbins=30,
            color="listing_status",
            title="Price Distribution",
            marginal="box",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.scatter(
            filtered,
            x="sqft",
            y="price",
            color="listing_status",
            size="bed",
            hover_data=["neighborhood", "title"],
            title="Price vs Size",
            trendline="ols",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Price per sqft by size band
    ppsf_band = (
        filtered.groupby("size_band", observed=True)["price_per_sqft"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    st.subheader("Price per sqft by Size Band")
    st.dataframe(
        ppsf_band.style.format({"mean": "${:,.0f}", "median": "${:,.0f}"}),
        use_container_width=True,
    )

with tab3:
    neigh_stats = (
        filtered.groupby("neighborhood")
        .agg(
            listings=("id", "count"),
            avg_price=("price", "mean"),
            median_price=("price", "median"),
            avg_ppsf=("price_per_sqft", "mean"),
            avg_sqft=("sqft", "mean"),
            pct_sold=("listing_status", lambda x: (x == "sold").mean() * 100),
        )
        .round(0)
        .sort_values("avg_price", ascending=False)
        .reset_index()
    )

    st.dataframe(
        neigh_stats.style.format({
            "avg_price": "${:,.0f}",
            "median_price": "${:,.0f}",
            "avg_ppsf": "${:,.0f}",
            "avg_sqft": "{:,.0f}",
            "pct_sold": "{:.1f}%",
        }),
        use_container_width=True,
    )

    fig = px.bar(
        neigh_stats,
        x="neighborhood",
        y="avg_ppsf",
        color="pct_sold",
        title="Avg $/sqft by Neighborhood (color = % sold)",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    # Fireplace premium
    fire_stats = (
        filtered.groupby("fire")
        .agg(
            n=("id", "count"),
            avg_price=("price", "mean"),
            avg_ppsf=("price_per_sqft", "mean"),
        )
        .round(0)
    )
    st.subheader("Fireplace Premium")
    st.dataframe(fire_stats.style.format({"avg_price": "${:,.0f}", "avg_ppsf": "${:,.0f}"}))

    # Renovation premium
    reno_stats = (
        filtered.groupby("renovated")
        .agg(
            n=("id", "count"),
            avg_price=("price", "mean"),
            avg_ppsf=("price_per_sqft", "mean"),
        )
        .round(0)
    )
    st.subheader("Renovation Premium")
    st.dataframe(reno_stats.style.format({"avg_price": "${:,.0f}", "avg_ppsf": "${:,.0f}"}))

    # Age vs price
    fig = px.scatter(
        filtered,
        x="age",
        y="price_per_sqft",
        color="renovated",
        title="Age vs $/sqft (renovated vs not)",
        trendline="ols",
    )
    st.plotly_chart(fig, use_container_width=True)


    st.divider()
st.subheader("Filtered Listings")
st.dataframe(
    filtered[
        ["id", "title", "neighborhood", "price", "bed", "bath", "sqft",
         "price_per_sqft", "y_built", "ren_y", "fire", "listing_status"]
    ].sort_values("price", ascending=False),
    use_container_width=True,
    height=400,
)

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", csv, "filtered_apartments.csv", "text/csv")


