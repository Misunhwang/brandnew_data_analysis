import io
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BrandNew GERD Explorer", layout="wide")

DEFAULT_FILE = "1.GERD.xlsx"
DEFAULT_SHEET = "1- CERD (USD applied)"
ANALYSIS_YEAR_MIN = 2015
ANALYSIS_YEAR_MAX = 2021


@st.cache_data
def load_excel(file_source, sheet_name: str) -> pd.DataFrame:
    """Load the uploaded Excel file or a local default file."""
    return pd.read_excel(file_source, sheet_name=sheet_name)


@st.cache_data
def prepare_gerd_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and keep the standard analysis period."""
    df = raw_df.rename(
        columns={
            "REF_AREA": "country_code",
            "Reference area": "country",
            "TIME_PERIOD": "year",
            "USD_value*milion": "gerd_usd",
        }
    ).copy()

    required_cols = ["country_code", "country", "year", "gerd_usd"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "The dataset does not contain the expected columns: "
            + ", ".join(missing_cols)
        )

    df = df[required_cols].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["gerd_usd"] = pd.to_numeric(df["gerd_usd"], errors="coerce")
    df = df.dropna(subset=["country", "year", "gerd_usd"])
    df["year"] = df["year"].astype(int)
    df = df[df["year"].between(ANALYSIS_YEAR_MIN, ANALYSIS_YEAR_MAX)].copy()
    df = df.sort_values(["country", "year"])
    return df


@st.cache_data
def calculate_summary(df: pd.DataFrame) -> dict:
    duplicates = df.duplicated(subset=["country", "year"]).sum()
    return {
        "rows": len(df),
        "countries": df["country"].nunique(),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "duplicates": int(duplicates),
        "missing_values": int(df.isna().sum().sum()),
    }


@st.cache_data
def compute_growth(df: pd.DataFrame) -> pd.DataFrame:
    growth_df = df.sort_values(["country", "year"]).copy()
    growth_df["growth_rate"] = (
        growth_df.groupby("country")["gerd_usd"].pct_change() * 100
    )
    return growth_df


@st.cache_data
def build_country_growth_summary(growth_df: pd.DataFrame) -> pd.DataFrame:
    valid = growth_df.dropna(subset=["growth_rate"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["country", "avg_growth_rate", "observations"])
    summary = (
        valid.groupby("country")
        .agg(
            avg_growth_rate=("growth_rate", "mean"),
            observations=("growth_rate", "count"),
        )
        .reset_index()
        .sort_values("avg_growth_rate", ascending=False)
    )
    return summary


def add_download_button(df: pd.DataFrame, label: str, filename: str) -> None:
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_data, file_name=filename, mime="text/csv")


st.title("BrandNew GERD Explorer")
st.caption(
    "Prototype Streamlit app for exploratory analysis of OECD GERD data "
    "(Gross Domestic Expenditure on R&D)."
)

st.markdown(
    """
This app helps the BrandNew team review the cleaned GERD dataset before merging it with
other indicators such as education expenditure, GDP, and outcome variables.

Suggested interpretation focus:
- **Data quality**: missing values, duplicates, country-year coverage
- **Distribution**: skewness, outliers, cross-country gaps
- **Trend**: how GERD changes over time
- **Growth**: how quickly each country's GERD changes from year to year
"""
)

with st.sidebar:
    st.header("Data Input")
    uploaded_file = st.file_uploader(
        "Upload the GERD Excel file", type=["xlsx", "xls"]
    )
    sheet_name = st.text_input("Sheet name", value=DEFAULT_SHEET)

    st.header("Filters")
    analysis_years = st.slider(
        "Analysis year range",
        min_value=2015,
        max_value=2023,
        value=(ANALYSIS_YEAR_MIN, ANALYSIS_YEAR_MAX),
        step=1,
    )

try:
    if uploaded_file is not None:
        raw_df = load_excel(uploaded_file, sheet_name)
        source_name = uploaded_file.name
    else:
        raw_df = load_excel(DEFAULT_FILE, sheet_name)
        source_name = DEFAULT_FILE
except Exception as exc:
    st.error(f"Could not load the Excel file: {exc}")
    st.stop()

try:
    df = prepare_gerd_data(raw_df)
except Exception as exc:
    st.error(f"Could not prepare the GERD dataset: {exc}")
    st.stop()

# Apply sidebar year filter after the standard preparation step.
df = df[df["year"].between(analysis_years[0], analysis_years[1])].copy()
if df.empty:
    st.warning("No data is available for the selected year range.")
    st.stop()

growth_df = compute_growth(df)
growth_summary = build_country_growth_summary(growth_df)
summary = calculate_summary(df)
year_counts = (
    df.groupby("year")["country"].nunique().reset_index(name="country_count")
)

country_options: List[str] = sorted(df["country"].unique().tolist())
default_countries = [
    country
    for country in ["Korea", "New Zealand", "United States", "Germany"]
    if country in country_options
]

with st.sidebar:
    selected_countries = st.multiselect(
        "Selected countries for trend and growth charts",
        options=country_options,
        default=default_countries[:4] if default_countries else country_options[:4],
    )
    selected_bar_year = st.selectbox(
        "Year for country ranking",
        options=sorted(df["year"].unique().tolist()),
        index=len(sorted(df["year"].unique().tolist())) - 1,
    )
    top_n = st.slider("Top N countries for ranking", 5, 20, 10, 1)

st.subheader("1. Cleaned Data Preview")
st.write(f"**Data source:** {source_name}")
st.dataframe(df.head(20), use_container_width=True)
add_download_button(df, "Download filtered GERD data as CSV", "gerd_filtered.csv")

st.subheader("2. Data Quality Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", summary["rows"])
col2.metric("Countries", summary["countries"])
col3.metric("Year Range", f"{summary['year_min']}–{summary['year_max']}")

col4, col5, col6 = st.columns(3)
col4.metric("Missing Values", summary["missing_values"])
col5.metric("Duplicate Country-Year Rows", summary["duplicates"])
col6.metric("Median GERD", f"{df['gerd_usd'].median():,.0f}")

st.markdown(
    """
**Interpretation**
- If missing values and duplicate country-year rows are zero, the file is in good condition for exploratory analysis.
- The year range and number of countries show how much cross-country and time-series coverage is available.
- The median is useful because GERD values are usually highly uneven across countries.
"""
)

st.write("**Country coverage by year**")
st.dataframe(year_counts, use_container_width=True)

st.subheader("3. Descriptive Statistics")
st.dataframe(df[["gerd_usd"]].describe().T, use_container_width=True)
st.markdown(
    """
**Interpretation**
- Compare the **mean** and the **median**. If the mean is much larger than the median, the distribution is likely right-skewed.
- A very large maximum compared with the lower quartiles suggests that a few countries contribute extremely high GERD values.
"""
)

st.subheader("4. Distribution Charts")
dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(df["gerd_usd"])
    ax.set_title("Boxplot of GERD")
    ax.set_ylabel("GERD (USD)")
    st.pyplot(fig)

with dist_col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["gerd_usd"], bins=30)
    ax.set_title("Histogram of GERD")
    ax.set_xlabel("GERD (USD)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

with st.expander("Optional: log-scale distribution"):
    positive_df = df[df["gerd_usd"] > 0].copy()
    if not positive_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(np.log10(positive_df["gerd_usd"]), bins=30)
        ax.set_title("Histogram of log10(GERD)")
        ax.set_xlabel("log10(GERD)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        st.markdown(
            "The log-scale view can make it easier to inspect small and large countries in the same chart."
        )

st.markdown(
    """
**Interpretation**
- The boxplot helps identify **outliers** and the overall spread of the distribution.
- The histogram shows whether GERD values are concentrated in the lower range while a small number of countries sit far above the rest.
- If that pattern appears, the dataset is not evenly distributed across countries.
"""
)

st.subheader("5. GERD Trend Over Time")
trend_type = st.radio(
    "Choose trend measure",
    options=["Average GERD", "Total GERD"],
    horizontal=True,
)

if trend_type == "Average GERD":
    yearly_trend = df.groupby("year")["gerd_usd"].mean().reset_index(name="gerd_value")
    y_label = "Average GERD (USD)"
else:
    yearly_trend = df.groupby("year")["gerd_usd"].sum().reset_index(name="gerd_value")
    y_label = "Total GERD (USD)"

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(yearly_trend["year"], yearly_trend["gerd_value"], marker="o")
ax.set_title(f"{trend_type} by Year")
ax.set_xlabel("Year")
ax.set_ylabel(y_label)
ax.grid(True)
st.pyplot(fig)

with st.expander("Distribution by year"):
    fig, ax = plt.subplots(figsize=(10, 5))
    df.boxplot(column="gerd_usd", by="year", ax=ax)
    ax.set_title("GERD Distribution by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("GERD (USD)")
    plt.suptitle("")
    st.pyplot(fig)

st.markdown(
    """
**Interpretation**
- The line chart shows whether GERD is generally rising, stable, or falling over time.
- Average GERD is often better for comparison when country coverage changes by year.
- Total GERD is useful for showing overall scale, but it can be influenced by different numbers of countries in different years.
"""
)

st.subheader("6. Country Ranking")
rank_df = (
    df[df["year"] == selected_bar_year]
    .sort_values("gerd_usd", ascending=False)
    .head(top_n)
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(rank_df["country"], rank_df["gerd_usd"])
ax.set_title(f"Top {top_n} Countries by GERD in {selected_bar_year}")
ax.set_xlabel("Country")
ax.set_ylabel("GERD (USD)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig)

st.dataframe(rank_df[["country", "year", "gerd_usd"]], use_container_width=True)

st.markdown(
    """
**Interpretation**
- This chart compares countries by absolute GERD in one selected year.
- It is useful for identifying the largest R&D spenders, but it should be interpreted as a **scale comparison**, not as a measure of efficiency or impact.
"""
)

st.subheader("7. Selected Country Trends")
if selected_countries:
    selected_df = df[df["country"].isin(selected_countries)].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for country in selected_countries:
        temp = selected_df[selected_df["country"] == country]
        ax.plot(temp["year"], temp["gerd_usd"], marker="o", label=country)
    ax.set_title("GERD Trend for Selected Countries")
    ax.set_xlabel("Year")
    ax.set_ylabel("GERD (USD)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        """
**Interpretation**
- This view makes it easier to compare individual national trajectories.
- It can show whether GERD growth is steady, volatile, or flat for selected countries.
- Large countries may visually dominate the chart, so growth-rate analysis is also useful.
"""
    )
else:
    st.info("Select at least one country in the sidebar to display the country trend chart.")

st.subheader("8. Growth Rate Analysis")
if selected_countries:
    selected_growth = growth_df[growth_df["country"].isin(selected_countries)].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for country in selected_countries:
        temp = selected_growth[selected_growth["country"] == country]
        ax.plot(temp["year"], temp["growth_rate"], marker="o", label=country)
    ax.axhline(0, linewidth=1)
    ax.set_title("GERD Growth Rate for Selected Countries")
    ax.set_xlabel("Year")
    ax.set_ylabel("Growth Rate (%)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Select at least one country in the sidebar to display the growth-rate chart.")

st.markdown(
    """
**Interpretation**
- Growth rate highlights the **speed of change**, not only the size of GERD.
- A country with smaller GERD can still show strong growth.
- Large swings suggest volatility, while smoother patterns suggest more stable year-to-year changes.
"""
)

st.write("**Average growth rate by country**")
st.dataframe(growth_summary, use_container_width=True)
add_download_button(growth_summary, "Download growth summary as CSV", "gerd_growth_summary.csv")

st.subheader("9. Suggested Insights for Team Discussion")
st.markdown(
    """
1. **Data readiness**: The cleaned GERD file can be used as a solid base dataset if country-year duplicates remain at zero.
2. **Distribution**: GERD values are typically highly uneven, so median and boxplot interpretation matter.
3. **Trend**: An overall upward pattern may suggest growing R&D investment over time.
4. **Country comparison**: Large economies usually dominate absolute GERD rankings.
5. **Growth**: Growth-rate analysis helps identify which countries are expanding GERD quickly, even if their total GERD is smaller.
6. **Next merge step**: The next stage is to combine GERD with education expenditure, GDP, and outcome indicators using a country-year key.
"""
)

st.subheader("10. Streamlit Notes")
st.code(
    "streamlit run app.py",
    language="bash",
)
st.caption(
    "If you deploy this app, keep `1.GERD.xlsx` in the same repository folder as `app.py`, "
    "or upload the file through the sidebar when running the app."
)
