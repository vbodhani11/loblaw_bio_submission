#!/usr/bin/env python3
"""Initialize the SQLite database and load cell-count.csv.

Run directly from the repository root:
    python load_data.py

No command-line arguments are required.
"""

from __future__ import annotations

import sqlite3
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "cell-count.csv"
DB_PATH = ROOT / "clinical_trial.db"
DATA_URL = "https://drive.google.com/uc?export=download&id=1eMfLCQBIqChy8FVej5yE-9h9UL7oTvVy"

POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]
METADATA_COLUMNS = [
    "project",
    "subject",
    "condition",
    "age",
    "sex",
    "treatment",
    "response",
    "sample",
    "sample_type",
    "time_from_treatment_start",
]
EXPECTED_COLUMNS = METADATA_COLUMNS + POPULATIONS

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE projects (
    project_id TEXT PRIMARY KEY
);

CREATE TABLE subjects (
    project_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    age INTEGER,
    sex TEXT,
    treatment TEXT,
    response TEXT,
    PRIMARY KEY (project_id, subject_id),
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);

CREATE TABLE samples (
    sample_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    sample_type TEXT NOT NULL,
    time_from_treatment_start INTEGER NOT NULL,
    FOREIGN KEY (project_id, subject_id)
        REFERENCES subjects(project_id, subject_id)
);

CREATE TABLE cell_populations (
    population_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE cell_counts (
    sample_id TEXT NOT NULL,
    population_id INTEGER NOT NULL,
    count INTEGER NOT NULL CHECK (count >= 0),
    PRIMARY KEY (sample_id, population_id),
    FOREIGN KEY (sample_id) REFERENCES samples(sample_id),
    FOREIGN KEY (population_id) REFERENCES cell_populations(population_id)
);

CREATE INDEX idx_subjects_trial_filters
    ON subjects(condition, treatment, response, sex);
CREATE INDEX idx_samples_subject
    ON samples(project_id, subject_id);
CREATE INDEX idx_samples_type_time
    ON samples(sample_type, time_from_treatment_start);
CREATE INDEX idx_cell_counts_population
    ON cell_counts(population_id);

CREATE VIEW sample_cell_frequencies AS
WITH counts_with_total AS (
    SELECT
        cc.sample_id,
        cp.name AS population,
        cc.count,
        SUM(cc.count) OVER (PARTITION BY cc.sample_id) AS total_count
    FROM cell_counts AS cc
    JOIN cell_populations AS cp
      ON cp.population_id = cc.population_id
)
SELECT
    sample_id AS sample,
    total_count,
    population,
    count,
    CASE
        WHEN total_count = 0 THEN NULL
        ELSE 100.0 * count / total_count
    END AS percentage
FROM counts_with_total;
"""


def _download_data_if_missing(csv_path: Path = CSV_PATH) -> None:
    """Download the official input file if it is not already in the repo.

    The assignment expects cell-count.csv in the repository. This fallback keeps
    Codespaces reproducible even if the file was accidentally omitted.
    """
    if csv_path.exists():
        return

    print(f"{csv_path.name} not found; attempting download from the provided Google Drive link...")
    tmp_path = csv_path.with_suffix(".download")
    try:
        request = urllib.request.Request(
            DATA_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            tmp_path.write_bytes(response.read())

        # Fail clearly if Google returns an HTML sign-in/interstitial page.
        first_line = tmp_path.open("r", encoding="utf-8", errors="ignore").readline().strip()
        if not first_line.startswith("project,subject,"):
            raise RuntimeError("Downloaded content was not the expected CSV file.")

        tmp_path.replace(csv_path)
        print(f"Downloaded {csv_path.name} successfully.")
    except Exception as exc:  # noqa: BLE001 - user-facing setup error
        tmp_path.unlink(missing_ok=True)
        raise FileNotFoundError(
            "cell-count.csv is missing and automatic download failed. "
            "Download the file from the assignment link and place it in the repository root, "
            "then rerun `python load_data.py`."
        ) from exc


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected CSV columns: {missing}")

    # Reject duplicate or unexpected aliases instead of silently guessing.
    df = df[EXPECTED_COLUMNS].copy()

    text_columns = [
        "project",
        "subject",
        "condition",
        "sex",
        "treatment",
        "response",
        "sample",
        "sample_type",
    ]
    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()

    # Normalize filterable categories while preserving IDs exactly.
    for column in ["condition", "treatment", "response", "sample_type"]:
        df[column] = df[column].str.lower()
    df["sex"] = df["sex"].str.upper()

    # Convert blank response values (e.g. healthy controls) to NULL.
    df["response"] = df["response"].replace({"": pd.NA, "<NA>": pd.NA})

    numeric_columns = ["age", "time_from_treatment_start"] + POPULATIONS
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    required_non_null = [
        "project",
        "subject",
        "condition",
        "sample",
        "sample_type",
        "time_from_treatment_start",
    ] + POPULATIONS
    null_counts = df[required_non_null].isna().sum()
    bad_nulls = null_counts[null_counts > 0]
    if not bad_nulls.empty:
        raise ValueError(f"Required columns contain missing values: {bad_nulls.to_dict()}")

    for column in ["time_from_treatment_start"] + POPULATIONS:
        values = df[column]
        if ((values % 1) != 0).any():
            raise ValueError(f"Column {column!r} must contain whole numbers.")
        df[column] = values.astype("int64")

    if (df[POPULATIONS] < 0).any().any():
        raise ValueError("Cell counts must be non-negative.")

    if df["sample"].duplicated().any():
        duplicates = df.loc[df["sample"].duplicated(), "sample"].head().tolist()
        raise ValueError(f"Sample IDs must be unique; duplicates include {duplicates}")

    # Each project/subject should have one coherent subject-level metadata record.
    subject_metadata = ["condition", "age", "sex", "treatment", "response"]
    uniqueness = (
        df.groupby(["project", "subject"], dropna=False)[subject_metadata]
        .nunique(dropna=False)
    )
    conflicts = uniqueness.gt(1).any(axis=1)
    if conflicts.any():
        first_conflict = conflicts[conflicts].index[0]
        raise ValueError(
            "Conflicting subject metadata found for project/subject "
            f"{first_conflict}."
        )

    return df


def _records(frame: pd.DataFrame, columns: list[str]) -> list[tuple]:
    """Convert a frame to SQLite-friendly tuples, replacing pandas NA with None."""
    subset = frame[columns].copy().astype(object)
    subset = subset.where(pd.notna(subset), None)
    return list(subset.itertuples(index=False, name=None))


def initialize_database(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH) -> None:
    """Create a fresh normalized SQLite database from the source CSV."""
    _download_data_if_missing(csv_path)

    raw = pd.read_csv(csv_path)
    df = _normalize_dataframe(raw)

    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)

        projects = df[["project"]].drop_duplicates().sort_values("project")
        conn.executemany(
            "INSERT INTO projects(project_id) VALUES (?)",
            _records(projects, ["project"]),
        )

        subject_columns = [
            "project",
            "subject",
            "condition",
            "age",
            "sex",
            "treatment",
            "response",
        ]
        subjects = df[subject_columns].drop_duplicates(["project", "subject"])
        conn.executemany(
            """
            INSERT INTO subjects(
                project_id, subject_id, condition, age, sex, treatment, response
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            _records(subjects, subject_columns),
        )

        sample_columns = [
            "sample",
            "project",
            "subject",
            "sample_type",
            "time_from_treatment_start",
        ]
        samples = df[sample_columns]
        conn.executemany(
            """
            INSERT INTO samples(
                sample_id, project_id, subject_id, sample_type, time_from_treatment_start
            ) VALUES (?, ?, ?, ?, ?)
            """,
            _records(samples, sample_columns),
        )

        population_rows = [(idx + 1, name) for idx, name in enumerate(POPULATIONS)]
        conn.executemany(
            "INSERT INTO cell_populations(population_id, name) VALUES (?, ?)",
            population_rows,
        )
        population_ids = {name: idx for idx, name in population_rows}

        long_counts = df[["sample"] + POPULATIONS].melt(
            id_vars="sample",
            var_name="population",
            value_name="count",
        )
        long_counts["population_id"] = long_counts["population"].map(population_ids)
        conn.executemany(
            "INSERT INTO cell_counts(sample_id, population_id, count) VALUES (?, ?, ?)",
            _records(long_counts, ["sample", "population_id", "count"]),
        )

        conn.commit()

        sample_count = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        measurement_count = conn.execute("SELECT COUNT(*) FROM cell_counts").fetchone()[0]
        subject_count = conn.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]
        project_count = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]

        print(f"Created {db_path.name}")
        print(f"  projects:     {project_count:,}")
        print(f"  subjects:     {subject_count:,}")
        print(f"  samples:      {sample_count:,}")
        print(f"  cell counts:  {measurement_count:,}")
    finally:
        conn.close()


def main() -> None:
    initialize_database()


if __name__ == "__main__":
    main()
