#!/usr/bin/env python3
"""Interactive Streamlit dashboard for Parts 2-4."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import (
    DB_PATH,
    POPULATION_ORDER,
    get_frequency_table,
    get_part3_cohort,
    get_part4_baseline_subset,
    get_part4_breakdowns,
    get_required_b_cell_answer,
    run_part3_statistics,
)

st.set_page_config(
    page_title="Loblaw Bio Immune Cell Analysis",
    page_icon="🧬",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_frequency_table(db_mtime: float) -> pd.DataFrame:
    del db_mtime
    return get_frequency_table()


@st.cache_data(show_spinner=False)
def load_part3(db_mtime: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    del db_mtime
    cohort = get_part3_cohort()
    return cohort, run_part3_statistics(cohort)


@st.cache_data(show_spinner=False)
def load_part4(db_mtime: float):
    del db_mtime
    baseline = get_part4_baseline_subset()
    breakdowns = get_part4_breakdowns(baseline)
    avg_b_cells, n_samples = get_required_b_cell_answer()
    return baseline, breakdowns, avg_b_cells, n_samples


def require_database() -> float:
    if not DB_PATH.exists():
        st.error(
            "`clinical_trial.db` was not found. Run `make pipeline` first, then restart the dashboard."
        )
        st.stop()
    return DB_PATH.stat().st_mtime


db_mtime = require_database()

st.title("🧬 Loblaw Bio — Immune Cell Clinical Trial Analysis")
st.caption("Interactive views for Parts 2–4, backed by the SQLite database created by the pipeline.")

# The prompt explicitly asks that quintazide be mentioned. Keep the mention honest
# rather than pretending it is present in the dataset.
with st.sidebar:
    st.header("Study notes")
    st.write("Primary response analysis: melanoma + miraclib + PBMC samples.")
    st.info(
        "Quintazide note: the assignment mentions quintazide, but the analysis does not "
        "invent or infer quintazide records. Only treatments actually present in the data are analyzed."
    )
    st.caption(f"Database: {Path(DB_PATH).name}")

part2_tab, part3_tab, part4_tab = st.tabs(
    ["Part 2 · Frequencies", "Part 3 · Response statistics", "Part 4 · Subsets"]
)

with part2_tab:
    st.subheader("Relative frequency of each immune cell population")
    frequencies = load_frequency_table(db_mtime)

    c1, c2 = st.columns([1, 2])
    with c1:
        selected_populations = st.multiselect(
            "Population",
            POPULATION_ORDER,
            default=POPULATION_ORDER,
        )
    with c2:
        sample_search = st.text_input("Sample contains", placeholder="e.g. sample00066")

    filtered = frequencies[frequencies["population"].isin(selected_populations)].copy()
    if sample_search.strip():
        filtered = filtered[
            filtered["sample"].str.contains(sample_search.strip(), case=False, na=False)
        ]

    st.metric("Rows shown", f"{len(filtered):,}")
    st.dataframe(filtered, use_container_width=True, height=430)
    st.download_button(
        "Download displayed Part 2 table",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="part2_frequency_table_filtered.csv",
        mime="text/csv",
    )

with part3_tab:
    st.subheader("Responders vs non-responders — melanoma + miraclib + PBMC")
    cohort, stats = load_part3(db_mtime)

    fig = px.box(
        cohort,
        x="population",
        y="percentage",
        color="response",
        category_orders={"population": POPULATION_ORDER, "response": ["yes", "no"]},
        labels={
            "population": "Immune cell population",
            "percentage": "Relative frequency (%)",
            "response": "Response",
        },
        points=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    significant = stats.loc[stats["significant_fdr_lt_0_05"], "population"].tolist()
    raw_significant = stats.loc[stats["significant_raw_p_lt_0_05"], "population"].tolist()

    a, b, c = st.columns(3)
    a.metric("Responder samples", int(stats["n_responders"].max()))
    b.metric("Non-responder samples", int(stats["n_nonresponders"].max()))
    c.metric("Significant after BH-FDR", len(significant))

    if significant:
        st.success("Significant after BH-FDR: " + ", ".join(significant))
    else:
        st.warning(
            "No population remains statistically significant after Benjamini-Hochberg "
            "correction at FDR < 0.05."
        )
        if raw_significant:
            st.caption(
                "At unadjusted p < 0.05, the following population(s) cross the nominal threshold: "
                + ", ".join(raw_significant)
                + ". The adjusted result is used for the main conclusion because five populations are tested."
            )

    display_stats = stats.copy()
    numeric_cols = [
        "median_responder_pct",
        "median_nonresponder_pct",
        "median_difference_pct_points",
        "p_value",
        "p_adjusted_bh",
        "rank_biserial_correlation",
    ]
    display_stats[numeric_cols] = display_stats[numeric_cols].round(5)
    st.dataframe(display_stats, use_container_width=True)
    st.caption(
        "Primary test: two-sided Mann-Whitney U. Multiple testing: Benjamini-Hochberg FDR "
        "across the five immune populations."
    )

with part4_tab:
    st.subheader("Baseline melanoma PBMC samples treated with miraclib")
    baseline, breakdowns, avg_b_cells, n_b_samples = load_part4(db_mtime)

    k1, k2, k3 = st.columns(3)
    k1.metric("Qualifying baseline samples", f"{baseline['sample'].nunique():,}")
    k2.metric("Unique subjects", f"{baseline.drop_duplicates(['project', 'subject']).shape[0]:,}")
    k3.metric("Projects represented", f"{baseline['project'].nunique():,}")

    left, middle, right = st.columns(3)
    with left:
        st.markdown("**Samples per project**")
        st.plotly_chart(
            px.bar(
                breakdowns["samples_per_project"],
                x="project",
                y="n_samples",
                text="n_samples",
            ),
            use_container_width=True,
        )
    with middle:
        st.markdown("**Subjects by response**")
        st.plotly_chart(
            px.pie(
                breakdowns["subjects_by_response"],
                names="response",
                values="n_subjects",
                hole=0.4,
            ),
            use_container_width=True,
        )
    with right:
        st.markdown("**Subjects by sex**")
        st.plotly_chart(
            px.pie(
                breakdowns["subjects_by_sex"],
                names="sex",
                values="n_subjects",
                hole=0.4,
            ),
            use_container_width=True,
        )

    st.markdown("### Required B-cell question")
    st.info(
        "This query intentionally uses **all sample types and all treatment types**. It filters only "
        "melanoma, male, responder, and time = 0, then averages the raw B-cell count."
    )
    q1, q2 = st.columns(2)
    q1.metric("Average B-cell count", f"{avg_b_cells:.2f}")
    q2.metric("Matching samples", f"{n_b_samples:,}")

    with st.expander("Show baseline subset rows"):
        st.dataframe(baseline, use_container_width=True, height=430)
