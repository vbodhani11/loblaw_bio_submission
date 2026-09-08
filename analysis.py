#!/usr/bin/env python3
"""Run Parts 2-4 of the Loblaw Bio clinical-trial analysis."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "clinical_trial.db"
OUTPUT_DIR = ROOT / "outputs"
POPULATION_ORDER = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]
ALPHA = 0.05


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run `python load_data.py` first."
        )
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def get_frequency_table(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Part 2: return one row per sample/population with relative frequency."""
    with connect(db_path) as conn:
        frame = pd.read_sql_query(
            """
            SELECT sample, total_count, population, count, percentage
            FROM sample_cell_frequencies
            ORDER BY sample, population
            """,
            conn,
        )
    return frame


def get_part3_cohort(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return melanoma + miraclib + PBMC frequencies for response comparison."""
    with connect(db_path) as conn:
        frame = pd.read_sql_query(
            """
            SELECT
                f.sample,
                sa.project_id AS project,
                sa.subject_id AS subject,
                su.response,
                f.population,
                f.count,
                f.total_count,
                f.percentage
            FROM sample_cell_frequencies AS f
            JOIN samples AS sa
              ON sa.sample_id = f.sample
            JOIN subjects AS su
              ON su.project_id = sa.project_id
             AND su.subject_id = sa.subject_id
            WHERE su.condition = 'melanoma'
              AND su.treatment = 'miraclib'
              AND sa.sample_type = 'pbmc'
              AND su.response IN ('yes', 'no')
            ORDER BY f.population, su.response, f.sample
            """,
            conn,
        )
    return frame


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p

    if not np.all(np.isfinite(p)):
        # A single NaN would otherwise poison every entry through
        # np.minimum.accumulate. Callers should avoid feeding NaN p-values,
        # but fail safe here rather than silently corrupting unrelated results.
        raise ValueError("benjamini_hochberg received non-finite p-value(s).")

    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def run_part3_statistics(cohort: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U tests with BH-FDR correction across five populations."""
    rows: list[dict] = []

    for population in POPULATION_ORDER:
        pop = cohort[cohort["population"] == population]
        responders = pop.loc[pop["response"] == "yes", "percentage"].dropna().to_numpy()
        nonresponders = pop.loc[pop["response"] == "no", "percentage"].dropna().to_numpy()

        if len(responders) == 0 or len(nonresponders) == 0:
            raise ValueError(f"Cannot test {population}: one response group has no observations.")

        test = mannwhitneyu(
            responders,
            nonresponders,
            alternative="two-sided",
            method="auto",
        )
        p_value = float(test.pvalue)
        if not np.isfinite(p_value):
            # scipy returns NaN when both groups are fully tied (zero variance,
            # e.g. every value identical). There is no evidence of a difference
            # in that case, so treat it conservatively as p = 1.0 instead of
            # letting NaN propagate through BH correction and poison every
            # other population's adjusted p-value.
            p_value = 1.0
        n_r = len(responders)
        n_nr = len(nonresponders)
        # Rank-biserial correlation, signed from responder perspective.
        rank_biserial = (2.0 * test.statistic / (n_r * n_nr)) - 1.0

        rows.append(
            {
                "population": population,
                "n_responders": n_r,
                "n_nonresponders": n_nr,
                "median_responder_pct": float(np.median(responders)),
                "median_nonresponder_pct": float(np.median(nonresponders)),
                "median_difference_pct_points": float(
                    np.median(responders) - np.median(nonresponders)
                ),
                "mann_whitney_u": float(test.statistic),
                "p_value": p_value,
                "rank_biserial_correlation": float(rank_biserial),
            }
        )

    results = pd.DataFrame(rows)
    results["p_adjusted_bh"] = benjamini_hochberg(results["p_value"].to_numpy())
    results["significant_raw_p_lt_0_05"] = results["p_value"] < ALPHA
    results["significant_fdr_lt_0_05"] = results["p_adjusted_bh"] < ALPHA
    return results.sort_values("p_value", ignore_index=True)


def save_part3_boxplot(cohort: pd.DataFrame, output_path: Path) -> None:
    """Create the assignment boxplot comparing responders and non-responders."""
    data = []
    labels = []
    for population in POPULATION_ORDER:
        pop = cohort[cohort["population"] == population]
        data.append(pop.loc[pop["response"] == "yes", "percentage"].dropna())
        labels.append(f"{population}\nResponder")
        data.append(pop.loc[pop["response"] == "no", "percentage"].dropna())
        labels.append(f"{population}\nNon-responder")

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Relative frequency (%)")
    ax.set_title("Melanoma PBMC samples receiving miraclib: responders vs non-responders")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def get_part4_baseline_subset(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Part 4.1: melanoma PBMC baseline samples treated with miraclib."""
    with connect(db_path) as conn:
        frame = pd.read_sql_query(
            """
            SELECT
                sa.sample_id AS sample,
                sa.project_id AS project,
                sa.subject_id AS subject,
                su.response,
                su.sex,
                su.age,
                su.condition,
                su.treatment,
                sa.sample_type,
                sa.time_from_treatment_start,
                MAX(CASE WHEN cp.name = 'b_cell' THEN cc.count END) AS b_cell,
                MAX(CASE WHEN cp.name = 'cd8_t_cell' THEN cc.count END) AS cd8_t_cell,
                MAX(CASE WHEN cp.name = 'cd4_t_cell' THEN cc.count END) AS cd4_t_cell,
                MAX(CASE WHEN cp.name = 'nk_cell' THEN cc.count END) AS nk_cell,
                MAX(CASE WHEN cp.name = 'monocyte' THEN cc.count END) AS monocyte
            FROM samples AS sa
            JOIN subjects AS su
              ON su.project_id = sa.project_id
             AND su.subject_id = sa.subject_id
            JOIN cell_counts AS cc
              ON cc.sample_id = sa.sample_id
            JOIN cell_populations AS cp
              ON cp.population_id = cc.population_id
            WHERE su.condition = 'melanoma'
              AND su.treatment = 'miraclib'
              AND sa.sample_type = 'pbmc'
              AND sa.time_from_treatment_start = 0
            GROUP BY
                sa.sample_id,
                sa.project_id,
                sa.subject_id,
                su.response,
                su.sex,
                su.age,
                su.condition,
                su.treatment,
                sa.sample_type,
                sa.time_from_treatment_start
            ORDER BY sa.project_id, sa.sample_id
            """,
            conn,
        )
    return frame


def get_part4_breakdowns(baseline: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Part 4.2: samples/project and unique subjects by response/sex."""
    samples_per_project = (
        baseline.groupby("project", as_index=False)
        .agg(n_samples=("sample", "nunique"))
        .sort_values("project")
    )

    subjects = baseline.drop_duplicates(["project", "subject"])
    subjects_by_response = (
        subjects.groupby("response", dropna=False, as_index=False)
        .agg(n_subjects=("subject", "size"))
        .sort_values("response", na_position="last")
    )
    subjects_by_sex = (
        subjects.groupby("sex", dropna=False, as_index=False)
        .agg(n_subjects=("subject", "size"))
        .sort_values("sex", na_position="last")
    )

    return {
        "samples_per_project": samples_per_project,
        "subjects_by_response": subjects_by_response,
        "subjects_by_sex": subjects_by_sex,
    }


def get_required_b_cell_answer(db_path: Path = DB_PATH) -> tuple[float, int]:
    """Final required question, intentionally NOT inheriting PBMC/miraclib filters.

    Scope from the prompt:
      - melanoma
      - male
      - responder
      - time = 0
      - all sample types
      - all treatment types
    """
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                AVG(cc.count) AS avg_b_cells,
                COUNT(*) AS n_matching_samples
            FROM cell_counts AS cc
            JOIN cell_populations AS cp
              ON cp.population_id = cc.population_id
            JOIN samples AS sa
              ON sa.sample_id = cc.sample_id
            JOIN subjects AS su
              ON su.project_id = sa.project_id
             AND su.subject_id = sa.subject_id
            WHERE su.condition = 'melanoma'
              AND su.sex = 'M'
              AND su.response = 'yes'
              AND sa.time_from_treatment_start = 0
              AND cp.name = 'b_cell'
            """
        ).fetchone()

    if row is None or row[0] is None:
        raise ValueError("No samples matched the final Part 4 B-cell query.")
    return round(float(row[0]), 2), int(row[1])


def write_summary_json(
    output_path: Path,
    stats: pd.DataFrame,
    baseline: pd.DataFrame,
    breakdowns: dict[str, pd.DataFrame],
    avg_b_cells: float,
    n_b_samples: int,
) -> None:
    significant = stats.loc[stats["significant_fdr_lt_0_05"], "population"].tolist()
    payload = {
        "part3": {
            "alpha": ALPHA,
            "test": "two-sided Mann-Whitney U",
            "multiple_testing": "Benjamini-Hochberg FDR",
            "significant_populations_after_fdr": significant,
        },
        "part4": {
            "baseline_subset_samples": int(baseline["sample"].nunique()),
            "samples_per_project": dict(
                zip(
                    breakdowns["samples_per_project"]["project"],
                    breakdowns["samples_per_project"]["n_samples"].astype(int),
                )
            ),
            "subjects_by_response": dict(
                zip(
                    breakdowns["subjects_by_response"]["response"].astype(str),
                    breakdowns["subjects_by_response"]["n_subjects"].astype(int),
                )
            ),
            "subjects_by_sex": dict(
                zip(
                    breakdowns["subjects_by_sex"]["sex"].astype(str),
                    breakdowns["subjects_by_sex"]["n_subjects"].astype(int),
                )
            ),
            "required_avg_b_cells_melanoma_male_responder_time0_all_types": avg_b_cells,
            "required_query_matching_samples": n_b_samples,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_pipeline(db_path: Path = DB_PATH, output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Part 2: calculating per-sample cell-population frequencies...")
    frequencies = get_frequency_table(db_path)
    frequencies.to_csv(output_dir / "part2_frequency_table.csv", index=False)

    print("Part 3: comparing melanoma/miraclib/PBMC responders vs non-responders...")
    cohort = get_part3_cohort(db_path)
    stats = run_part3_statistics(cohort)
    stats.to_csv(output_dir / "part3_stats.csv", index=False)
    save_part3_boxplot(cohort, output_dir / "part3_boxplot.png")

    print("Part 4: querying the baseline melanoma/miraclib/PBMC subset...")
    baseline = get_part4_baseline_subset(db_path)
    baseline.to_csv(output_dir / "part4_baseline_subset.csv", index=False)
    breakdowns = get_part4_breakdowns(baseline)
    for name, frame in breakdowns.items():
        frame.to_csv(output_dir / f"part4_{name}.csv", index=False)

    avg_b_cells, n_b_samples = get_required_b_cell_answer(db_path)
    answer_text = (
        "Required Part 4 question\n"
        "------------------------\n"
        "Considering melanoma males across ALL sample types and ALL treatment types,\n"
        "for responders at time_from_treatment_start = 0:\n"
        f"Average B-cell count = {avg_b_cells:.2f}\n"
        f"Matching samples = {n_b_samples}\n"
    )
    (output_dir / "part4_required_answer.txt").write_text(answer_text, encoding="utf-8")

    write_summary_json(
        output_dir / "analysis_summary.json",
        stats,
        baseline,
        breakdowns,
        avg_b_cells,
        n_b_samples,
    )

    significant = stats.loc[stats["significant_fdr_lt_0_05"], "population"].tolist()
    print(f"  Part 2 rows: {len(frequencies):,}")
    print(f"  Part 3 cohort rows (population-level): {len(cohort):,}")
    print(
        "  Significant populations after BH-FDR: "
        + (", ".join(significant) if significant else "none")
    )
    print(f"  Part 4 baseline samples: {len(baseline):,}")
    print(f"  Required average B-cell count: {avg_b_cells:.2f} (n={n_b_samples})")
    print(f"Outputs written to: {output_dir}")


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
