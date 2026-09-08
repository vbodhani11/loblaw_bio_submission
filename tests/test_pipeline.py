from pathlib import Path

import numpy as np

from analysis import (
    get_frequency_table,
    get_part3_cohort,
    get_part4_baseline_subset,
    get_part4_breakdowns,
    get_required_b_cell_answer,
    run_part3_statistics,
)
from load_data import initialize_database

FIXTURE = Path(__file__).parent / "fixtures" / "tiny-cell-count.csv"


def build_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    initialize_database(FIXTURE, db)
    return db


def test_frequency_percentages_sum_to_100(tmp_path):
    db = build_db(tmp_path)
    frequencies = get_frequency_table(db)
    sums = frequencies.groupby("sample")["percentage"].sum().to_numpy()
    assert np.allclose(sums, 100.0)
    assert list(frequencies.columns) == [
        "sample",
        "total_count",
        "population",
        "count",
        "percentage",
    ]


def test_part3_scope_and_statistics(tmp_path):
    db = build_db(tmp_path)
    cohort = get_part3_cohort(db)
    # a, b (responder miraclib PBMC) and c (non-responder miraclib PBMC), 5 populations each
    assert cohort["sample"].nunique() == 3
    stats = run_part3_statistics(cohort)
    assert set(stats["population"]) == {
        "b_cell",
        "cd8_t_cell",
        "cd4_t_cell",
        "nk_cell",
        "monocyte",
    }
    assert ((stats["p_adjusted_bh"] >= 0) & (stats["p_adjusted_bh"] <= 1)).all()


def test_part4_baseline_subset(tmp_path):
    db = build_db(tmp_path)
    baseline = get_part4_baseline_subset(db)
    assert set(baseline["sample"]) == {"a", "c"}
    breakdowns = get_part4_breakdowns(baseline)
    assert breakdowns["samples_per_project"]["n_samples"].sum() == 2


def test_required_b_cell_query_does_not_inherit_pbmc_or_miraclib_filters(tmp_path):
    db = build_db(tmp_path)
    average, n = get_required_b_cell_answer(db)
    # Includes sample a (miraclib/PBMC, B=10) and d (phauximab/TUMOR, B=30).
    assert average == 20.00
    assert n == 2
