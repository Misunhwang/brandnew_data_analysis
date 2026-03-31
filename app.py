from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Streamlit page config
# ============================================================
st.set_page_config(page_title="BrandNew GERD Dashboard", layout="wide")

# ============================================================
# Default workbook / sheet names
# ============================================================
DEFAULT_FILE = "all_data.xlsx"
GERD_SHEET = "1- CERD (USD applied)"
GERD_PISA_SHEET = "2. GERD+PISA"
ANALYSIS_YEAR_MIN = 2015
ANALYSIS_YEAR_MAX = 2023


# ============================================================
# Utility functions
# ============================================================
def add_download_button(df: pd.DataFrame, label: str, filename: str) -> None:
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def safe_log(series: pd.Series) -> pd.Series:
    return np.where(series > 0, np.log(series), np.nan)


@st.cache_data
# Load one workbook and read both sheets.
def load_workbook(file_source) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gerd_df = pd.read_excel(file_source, sheet_name=GERD_SHEET)

    # Re-open for second sheet if uploaded object is used.
    if hasattr(file_source, "seek"):
        file_source.seek(0)

    gerd_pisa_df = pd.read_excel(file_source, sheet_name=GERD_PISA_SHEET)
    return gerd_df, gerd_pisa_df


@st.cache_data
def prepare_gerd_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.rename(
        columns={
            "REF_AREA": "country_code",
            "Reference area": "country",
            "TIME_PERIOD": "year",
            "USD_value*milion": "gerd_usd",
            "USD_value*million": "gerd_usd",
        }
    ).copy()

    required_cols = ["country_code", "country", "year", "gerd_usd"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "The GERD sheet is missing expected columns: " + ", ".join(missing_cols)
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
def prepare_gerd_pisa_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.rename(
        columns={
            "REF_AREA": "country_code",
            "Reference area": "country",
            "TIME_PERIOD": "year",
            "USD_value*milion": "gerd_usd",
            "USD_value*million": "gerd_usd",
            "PISA_score": "pisa_score",
        }
    ).copy()

    required_cols = ["country_code", "country", "year", "gerd_usd", "pisa_score"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "The GERD+PISA sheet is missing expected columns: " + ", ".join(missing_cols)
        )

    df = df[required_cols].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["gerd_usd"] = pd.to_numeric(df["gerd_usd"], errors="coerce")
    df["pisa_score"] = pd.to_numeric(df["pisa_score"], errors="coerce")
    df = df.dropna(subset=["country", "year", "gerd_usd", "pisa_score"])
    df["year"] = df["year"].astype(int)
    df = df[df["year"].between(ANALYSIS_YEAR_MIN, ANALYSIS_YEAR_MAX)].copy()
    df["log_gerd"] = safe_log(df["gerd_usd"])
    df = df.sort_values(["country", "year"])
    return df


@st.cache_data
def calculate_summary(df: pd.DataFrame, value_col: str, dataset_name: str) -> dict:
    duplicates = int(df.duplicated(subset=["country", "year"]).sum())
    return {
        "dataset": dataset_name,
        "rows": len(df),
        "countries": int(df["country"].nunique()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "duplicates": duplicates,
        "missing_values": int(df.isna().sum().sum()),
        "median_value": float(df[value_col].median()),
    }


@st.cache_data
def compute_growth(df: pd.DataFrame) -> pd.DataFrame:
    growth_df = df.sort_values(["country", "year"]).copy()
    growth_df["growth_rate"] = growth_df.groupby("country")["gerd_usd"].pct_change() * 100
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


# ============================================================
# Sidebar
# ============================================================
st.title("BrandNew GERD Dashboard")
st.caption(
    "One-file Streamlit app using a single Excel workbook with two sheets: "
    "GERD-only and GERD+PISA."
)

with st.sidebar:
    st.header("Workbook Input")
    uploaded_file = st.file_uploader(
        "Upload all_data.xlsx",
        type=["xlsx", "xls"],
        help="Workbook must include both sheets: '1- CERD (USD applied)' and '2. GERD+ PISA'.",
    )

    st.header("Global Filters")
    analysis_years = st.slider(
        "Analysis year range",
        min_value=ANALYSIS_YEAR_MIN,
        max_value=ANALYSIS_YEAR_MAX,
        value=(ANALYSIS_YEAR_MIN, ANALYSIS_YEAR_MAX),
        step=1,
    )

# ============================================================
# Load workbook
# ============================================================
try:
    if uploaded_file is not None:
        raw_gerd_df, raw_gerd_pisa_df = load_workbook(uploaded_file)
        source_name = uploaded_file.name
    else:
        raw_gerd_df, raw_gerd_pisa_df = load_workbook(DEFAULT_FILE)
        source_name = DEFAULT_FILE
except Exception as exc:
    st.error(
        "Could not load the workbook. Make sure the Excel file is named 'all_data.xlsx' "
        f"or upload it manually, and confirm both sheet names are correct.\n\nError: {exc}"
    )
    st.stop()

try:
    gerd_df = prepare_gerd_data(raw_gerd_df)
    gerd_pisa_df = prepare_gerd_pisa_data(raw_gerd_pisa_df)
except Exception as exc:
    st.error(f"Could not prepare the datasets: {exc}")
    st.stop()

# Apply global year filter.
gerd_df = gerd_df[gerd_df["year"].between(analysis_years[0], analysis_years[1])].copy()
gerd_pisa_df = gerd_pisa_df[
    gerd_pisa_df["year"].between(analysis_years[0], analysis_years[1])
].copy()

if gerd_df.empty or gerd_pisa_df.empty:
    st.warning("No data is available for the selected year range.")
    st.stop()

# Precompute datasets.
gerd_growth_df = compute_growth(gerd_df)
gerd_growth_summary = build_country_growth_summary(gerd_growth_df)
gerd_summary = calculate_summary(gerd_df, "gerd_usd", "GERD Only")
gerd_pisa_summary = calculate_summary(gerd_pisa_df, "pisa_score", "GERD + PISA")

# Sidebar dataset-specific controls.
with st.sidebar:
    st.header("GERD Controls")
    gerd_country_options: List[str] = sorted(gerd_df["country"].unique().tolist())
    gerd_default_countries = [
        c
        for c in ["Korea", "New Zealand", "United States", "Germany"]
        if c in gerd_country_options
    ]
    selected_gerd_countries = st.multiselect(
        "Countries for GERD trend/growth",
        options=gerd_country_options,
        default=gerd_default_countries[:4] if gerd_default_countries else gerd_country_options[:4],
    )
    selected_bar_year = st.selectbox(
        "GERD ranking year",
        options=sorted(gerd_df["year"].unique().tolist()),
        index=len(sorted(gerd_df["year"].unique().tolist())) - 1,
    )
    top_n = st.slider("Top N countries", 5, 20, 10, 1)

    st.header("GERD + PISA Controls")
    pisa_year_options = sorted(gerd_pisa_df["year"].dropna().unique().tolist())
    selected_pisa_years = st.multiselect(
        "Years for GERD + PISA",
        pisa_year_options,
        default=pisa_year_options,
    )
    pisa_filtered_df = gerd_pisa_df[gerd_pisa_df["year"].isin(selected_pisa_years)].copy()
    pisa_country_options = sorted(pisa_filtered_df["country"].dropna().unique().tolist())
    selected_pisa_countries = st.multiselect(
        "Countries for PISA comparison",
        pisa_country_options,
        default=pisa_country_options[: min(6, len(pisa_country_options))],
    )

if pisa_filtered_df.empty:
    st.warning("No GERD + PISA data is available for the selected PISA years.")
    st.stop()

# ============================================================
# Main tabs
# ============================================================
tab_overview, tab_gerd, tab_gerd_pisa = st.tabs(
    ["Overview", "GERD Analysis", "GERD + PISA Analysis"]
)

# ============================================================
# Overview tab
# ============================================================
with tab_overview:
    st.subheader("Workbook Overview")
    st.write(f"**Data source:** {source_name}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Sheet 1: GERD Only")
        st.metric("Rows", gerd_summary["rows"])
        st.metric("Countries", gerd_summary["countries"])
        st.metric("Year Range", f"{gerd_summary['year_min']}–{gerd_summary['year_max']}")
        st.metric("Duplicates", gerd_summary["duplicates"])

    with col2:
        st.markdown("### Sheet 2: GERD + PISA")
        st.metric("Rows", gerd_pisa_summary["rows"])
        st.metric("Countries", gerd_pisa_summary["countries"])
        st.metric(
            "Year Range",
            f"{gerd_pisa_summary['year_min']}–{gerd_pisa_summary['year_max']}",
        )
        st.metric("Duplicates", gerd_pisa_summary["duplicates"])

    st.markdown(
        """
### How this app is organized
- **GERD Analysis** tab: explores GERD distribution, trend, country ranking, and growth.
- **GERD + PISA Analysis** tab: explores the relationship between GERD and PISA score.
- This version needs only **one `app.py`** and **one Excel file**.
        """
    )

    st.markdown("### Sheet previews")
    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        st.write(f"**{GERD_SHEET}**")
        st.dataframe(gerd_df.head(10), use_container_width=True)
    with preview_col2:
        st.write(f"**{GERD_PISA_SHEET}**")
        st.dataframe(gerd_pisa_df.head(10), use_container_width=True)

# ============================================================
# GERD Analysis tab
# ============================================================
with tab_gerd:
    st.header("GERD Analysis")
    st.caption("Exploratory analysis of the GERD-only sheet.")

    st.subheader("1. Cleaned Data Preview")
    st.dataframe(gerd_df.head(20), use_container_width=True)
    add_download_button(gerd_df, "Download filtered GERD data as CSV", "gerd_filtered.csv")

    st.subheader("2. Data Quality Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", gerd_summary["rows"])
    c2.metric("Countries", gerd_summary["countries"])
    c3.metric("Year Range", f"{gerd_summary['year_min']}–{gerd_summary['year_max']}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Missing Values", gerd_summary["missing_values"])
    c5.metric("Duplicate Country-Year Rows", gerd_summary["duplicates"])
    c6.metric("Median GERD", f"{gerd_df['gerd_usd'].median():,.0f}")

    year_counts = gerd_df.groupby("year")["country"].nunique().reset_index(name="country_count")
    st.write("**Country coverage by year**")
    st.dataframe(year_counts, use_container_width=True)

    st.subheader("3. Descriptive Statistics")
    st.dataframe(gerd_df[["gerd_usd"]].describe().T, use_container_width=True)

    st.subheader("4. Distribution Charts")
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(gerd_df["gerd_usd"])
        ax.set_title("Boxplot of GERD")
        ax.set_ylabel("GERD (USD)")
        st.pyplot(fig)
    with dist_col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(gerd_df["gerd_usd"], bins=30)
        ax.set_title("Histogram of GERD")
        ax.set_xlabel("GERD (USD)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with st.expander("Optional: log-scale distribution"):
        positive_df = gerd_df[gerd_df["gerd_usd"] > 0].copy()
        if not positive_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(np.log10(positive_df["gerd_usd"]), bins=30)
            ax.set_title("Histogram of log10(GERD)")
            ax.set_xlabel("log10(GERD)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    st.subheader("5. GERD Distribution by Year")
    fig, ax = plt.subplots(figsize=(10, 6))
    gerd_df.boxplot(column="gerd_usd", by="year", ax=ax)
    ax.set_title("GERD Distribution by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("GERD (USD)")
    plt.suptitle("")
    st.pyplot(fig)

    st.subheader("6. GERD Trend Over Time")
    trend_type = st.radio(
        "Choose trend measure",
        options=["Average GERD", "Total GERD"],
        horizontal=True,
        key="gerd_trend_type",
    )

    if trend_type == "Average GERD":
        yearly_trend = gerd_df.groupby("year")["gerd_usd"].mean().reset_index(name="gerd_value")
        y_label = "Average GERD (USD)"
    else:
        yearly_trend = gerd_df.groupby("year")["gerd_usd"].sum().reset_index(name="gerd_value")
        y_label = "Total GERD (USD)"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(yearly_trend["year"], yearly_trend["gerd_value"], marker="o")
    ax.set_title(f"{trend_type} by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("7. Country Ranking")
    rank_df = (
        gerd_df[gerd_df["year"] == selected_bar_year]
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

    st.subheader("8. Selected Country Trends")
    if selected_gerd_countries:
        selected_df = gerd_df[gerd_df["country"].isin(selected_gerd_countries)].copy()
        fig, ax = plt.subplots(figsize=(10, 5))
        for country in selected_gerd_countries:
            temp = selected_df[selected_df["country"] == country]
            ax.plot(temp["year"], temp["gerd_usd"], marker="o", label=country)
        ax.set_title("GERD Trend for Selected Countries")
        ax.set_xlabel("Year")
        ax.set_ylabel("GERD (USD)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Select at least one country in the sidebar to display the GERD trend chart.")

    st.subheader("9. Growth Rate Analysis")
    if selected_gerd_countries:
        selected_growth = gerd_growth_df[gerd_growth_df["country"].isin(selected_gerd_countries)].copy()
        fig, ax = plt.subplots(figsize=(10, 5))
        for country in selected_gerd_countries:
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
        st.info("Select at least one country in the sidebar to display the GERD growth chart.")

    st.write("**Average growth rate by country**")
    st.dataframe(gerd_growth_summary, use_container_width=True)
    add_download_button(
        gerd_growth_summary,
        "Download growth summary as CSV",
        "gerd_growth_summary.csv",
    )

# ============================================================
# GERD + PISA Analysis tab
# ============================================================
with tab_gerd_pisa:
    st.header("GERD + PISA Analysis")
    st.caption(
        "Exploratory analysis of the relationship between GERD and overall PISA score."
    )

    subt1, subt2, subt3, subt4, subt5, subt6 = st.tabs(
        [
            "Data Preview",
            "Data Quality",
            "PISA EDA",
            "GERD EDA",
            "GERD vs PISA",
            "Country Comparison",
        ]
    )

    with subt1:
        st.subheader("Dataset Preview")
        st.dataframe(pisa_filtered_df, use_container_width=True)
        st.write("Rows:", len(pisa_filtered_df))
        st.write("Countries:", pisa_filtered_df["country"].nunique())
        st.write("Years:", sorted(pisa_filtered_df["year"].dropna().unique().tolist()))
        add_download_button(
            pisa_filtered_df,
            "Download GERD + PISA filtered data as CSV",
            "gerd_pisa_filtered.csv",
        )

    with subt2:
        st.subheader("Data Quality Check")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(pisa_filtered_df))
        c2.metric("Countries", pisa_filtered_df["country"].nunique())
        c3.metric(
            "Duplicate country-year rows",
            int(pisa_filtered_df.duplicated(subset=["country", "year"]).sum()),
        )

        st.markdown("**Missing values by column**")
        st.dataframe(pisa_filtered_df.isna().sum().rename("missing_values").to_frame())

        st.markdown("**Data types**")
        dtypes_df = pd.DataFrame(
            {
                "column": pisa_filtered_df.columns,
                "dtype": pisa_filtered_df.dtypes.astype(str).values,
            }
        )
        st.dataframe(dtypes_df, use_container_width=True)

    with subt3:
        st.subheader("PISA Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.hist(pisa_filtered_df["pisa_score"].dropna(), bins=10)
            ax.set_title("Distribution of Overall PISA Score")
            ax.set_xlabel("PISA Score")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.boxplot(pisa_filtered_df["pisa_score"].dropna())
            ax.set_title("Boxplot of Overall PISA Score")
            ax.set_ylabel("PISA Score")
            st.pyplot(fig)

        if pisa_filtered_df["year"].nunique() > 1:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            pisa_filtered_df.boxplot(column="pisa_score", by="year", ax=ax)
            ax.set_title("PISA Score Distribution by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("PISA Score")
            plt.suptitle("")
            st.pyplot(fig)

    with subt4:
        st.subheader("GERD Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.hist(pisa_filtered_df["gerd_usd"].dropna(), bins=10)
            ax.set_title("Distribution of GERD")
            ax.set_xlabel("GERD (USD)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.hist(pisa_filtered_df["log_gerd"].dropna(), bins=10)
            ax.set_title("Log Distribution of GERD")
            ax.set_xlabel("log(GERD)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.boxplot(pisa_filtered_df["gerd_usd"].dropna())
            ax.set_title("Boxplot of GERD")
            ax.set_ylabel("GERD (USD)")
            st.pyplot(fig)
        with col4:
            if pisa_filtered_df["year"].nunique() > 1:
                fig, ax = plt.subplots(figsize=(8, 4.8))
                pisa_filtered_df.boxplot(column="gerd_usd", by="year", ax=ax)
                ax.set_title("GERD Distribution by Year")
                ax.set_xlabel("Year")
                ax.set_ylabel("GERD (USD)")
                plt.suptitle("")
                st.pyplot(fig)

    with subt5:
        st.subheader("Relationship Between GERD and PISA")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4.8))
            ax.scatter(pisa_filtered_df["gerd_usd"], pisa_filtered_df["pisa_score"])
            ax.set_title("GERD vs PISA Score")
            ax.set_xlabel("GERD (USD)")
            ax.set_ylabel("PISA Score")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4.8))
            if pisa_filtered_df["year"].nunique() > 1:
                for year in sorted(pisa_filtered_df["year"].dropna().unique()):
                    temp = pisa_filtered_df[pisa_filtered_df["year"] == year]
                    ax.scatter(temp["log_gerd"], temp["pisa_score"], label=str(int(year)))
                ax.legend(title="Year")
            else:
                ax.scatter(pisa_filtered_df["log_gerd"], pisa_filtered_df["pisa_score"])
            ax.set_title("log(GERD) vs PISA Score")
            ax.set_xlabel("log(GERD)")
            ax.set_ylabel("PISA Score")
            st.pyplot(fig)

        corr_raw = pisa_filtered_df[["gerd_usd", "pisa_score"]].corr().iloc[0, 1]
        corr_log = pisa_filtered_df[["log_gerd", "pisa_score"]].corr().iloc[0, 1]
        mc1, mc2 = st.columns(2)
        mc1.metric(
            "Correlation: GERD vs PISA",
            f"{corr_raw:.3f}" if pd.notna(corr_raw) else "N/A",
        )
        mc2.metric(
            "Correlation: log(GERD) vs PISA",
            f"{corr_log:.3f}" if pd.notna(corr_log) else "N/A",
        )

        valid = pisa_filtered_df[["log_gerd", "pisa_score"]].dropna()
        if len(valid) >= 2:
            x = valid["log_gerd"]
            y = valid["pisa_score"]
            coef = np.polyfit(x, y, 1)
            line = coef[0] * x + coef[1]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(x, y)
            ax.plot(x, line)
            ax.set_title("Regression: log(GERD) vs PISA Score")
            ax.set_xlabel("log(GERD)")
            ax.set_ylabel("PISA Score")
            st.pyplot(fig)

            st.write(f"**Slope:** {coef[0]:.3f}")
            st.write(f"**Intercept:** {coef[1]:.3f}")

    with subt6:
        st.subheader("Country Comparison")
        if not selected_pisa_countries:
            st.info("Select at least one country in the sidebar.")
        else:
            comp_df = pisa_filtered_df[pisa_filtered_df["country"].isin(selected_pisa_countries)].copy()

            fig, ax = plt.subplots(figsize=(10, 5))
            for country in selected_pisa_countries:
                temp = comp_df[comp_df["country"] == country].sort_values("year")
                ax.plot(temp["year"], temp["pisa_score"], marker="o", label=f"{country} - PISA")
            ax.set_title("PISA Score by Country")
            ax.set_xlabel("Year")
            ax.set_ylabel("PISA Score")
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(10, 5))
            for country in selected_pisa_countries:
                temp = comp_df[comp_df["country"] == country].sort_values("year")
                ax.plot(temp["year"], temp["gerd_usd"], marker="o", label=f"{country} - GERD")
            ax.set_title("GERD by Country")
            ax.set_xlabel("Year")
            ax.set_ylabel("GERD (USD)")
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            st.pyplot(fig)

st.divider()
st.subheader("Run Command")
st.code("streamlit run app.py", language="bash")
st.caption(
    "Deploy this app with `app.py` and one workbook named `all_data.xlsx` in the same folder."
)
