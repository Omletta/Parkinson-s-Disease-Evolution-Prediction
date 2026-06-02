"""Offline mock time-series API helpers (mirrors Kaggle amp_pd_peptide flow)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DATA_DIR, MOCK_DIR, MODEL_DIR, UPDRS_TARGETS
from .features import FeatureBuilder
from .metrics import masked_mae, masked_rmse
from .model import ProgressionModel
from .split import PatientSplit, filter_by_patients, split_patients_with_proteins

OFFSETS = [6, 12, 24]


def make_prediction_id(
    patient_id: int,
    visit_month: int,
    updrs_part: int,
    month_offset: int,
) -> str:
    """Format compatible with baseline notebook parsing (updrs at index 3, offset at 5)."""
    return f"{patient_id}_{visit_month}_0_{updrs_part}_0_{month_offset}"


def parse_prediction_id(prediction_id: str) -> dict:
    parts = prediction_id.split("_")
    return {
        "patient_id": int(parts[0]),
        "visit_month": int(parts[1]),
        "updrs_part": int(parts[3]),
        "month_offset": int(parts[5]),
        "prediction_visit_month": int(parts[1]) + int(parts[5]),
    }


def build_mock_api_files(
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: Path = MOCK_DIR,
) -> PatientSplit:
    """
    Build CSVs that mimic Kaggle example_test_files + sample_submission.

    Test patients are written to mock_api/; train your model on the remaining patients
    (or use --full-train and accept in-sample mock scores).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clinical = pd.read_csv(DATA_DIR / "train_clinical_data.csv")
    proteins = pd.read_csv(DATA_DIR / "train_proteins.csv")
    peptides = pd.read_csv(DATA_DIR / "train_peptides.csv")

    patient_split = split_patients_with_proteins(
        proteins, test_size=test_size, random_state=random_state
    )
    patient_split.save(output_dir / "patient_split.json")

    test_patients = patient_split.test_patient_set
    clin_test = filter_by_patients(clinical, test_patients)
    prot_test = filter_by_patients(proteins, test_patients)
    pep_test = filter_by_patients(peptides, test_patients)

    clin_api = clin_test.drop(columns=UPDRS_TARGETS, errors="ignore")
    clin_api.to_csv(output_dir / "test_clinical.csv", index=False)
    prot_test.to_csv(output_dir / "test_proteins.csv", index=False)
    pep_test.to_csv(output_dir / "test_peptides.csv", index=False)

    submission_rows = []
    answer_rows = []
    for _, row in clin_test.dropna(subset=["visit_id"]).iterrows():
        if row["visit_id"] not in set(prot_test["visit_id"]):
            continue
        patient_id = int(row["patient_id"])
        visit_month = int(row["visit_month"])
        for updrs_part in range(1, 5):
            for offset in OFFSETS:
                pid = make_prediction_id(patient_id, visit_month, updrs_part, offset)
                submission_rows.append(
                    {
                        "patient_id": patient_id,
                        "prediction_id": pid,
                        "rating": np.nan,
                    }
                )
                target_col = f"updrs_{updrs_part}"
                target_month = visit_month + offset
                true_row = clin_test[
                    (clin_test["patient_id"] == patient_id)
                    & (clin_test["visit_month"] == target_month)
                ]
                rating = np.nan
                if len(true_row) and pd.notna(true_row.iloc[0][target_col]):
                    rating = float(true_row.iloc[0][target_col])
                answer_rows.append(
                    {
                        "prediction_id": pid,
                        "rating": rating,
                        "patient_id": patient_id,
                        "updrs_part": updrs_part,
                        "month_offset": offset,
                    }
                )

    pd.DataFrame(submission_rows).to_csv(
        output_dir / "sample_submission.csv", index=False
    )
    pd.DataFrame(answer_rows).to_csv(output_dir / "answer_key.csv", index=False)
    return patient_split


def make_mock_env(mock_dir: Path = MOCK_DIR):
    """Configure public_timeseries_testing_util.MockApi."""
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from public_timeseries_testing_util import MockApi

    env = MockApi()
    env.input_paths = [
        str(mock_dir / "test_clinical.csv"),
        str(mock_dir / "test_peptides.csv"),
        str(mock_dir / "test_proteins.csv"),
        str(mock_dir / "sample_submission.csv"),
    ]
    env.group_id_column = "patient_id"
    env.export_group_id_column = True
    return env


def _predict_batch(
    model: ProgressionModel,
    clinical_history: pd.DataFrame,
    df_test: pd.DataFrame,
    df_proteins: pd.DataFrame,
) -> dict[tuple[int, int], dict[str, float]]:
    """Predict UPDRS at current CSF visits in this patient batch."""
    if df_proteins.empty:
        return {}

    x, _ = model.feature_builder.transform(clinical_history, df_proteins)
    preds = model.predict(x)
    visit_lookup = {}
    for visit_id in preds.index:
        visit_month = int(df_test.loc[df_test["visit_id"] == visit_id, "visit_month"].iloc[0])
        visit_lookup[visit_month] = preds.loc[visit_id].to_dict()
    return visit_lookup


def run_mock_api_inference(
    model: ProgressionModel,
    clinical_history: pd.DataFrame,
    mock_dir: Path = MOCK_DIR,
    submission_path: Path | None = None,
) -> pd.DataFrame:
    """
    Drive MockApi: one batch per patient_id, fill submission ratings.

    Uses current-visit model outputs for all horizon offsets (naive baseline).
    """
    env = make_mock_env(mock_dir)
    submission_path = submission_path or Path("submission.csv")

    for batch in env.iter_test():
        if batch is None:
            continue
        df_test, df_peptides, df_proteins, df_submission = batch

        visit_preds = _predict_batch(model, clinical_history, df_test, df_proteins)

        ratings = []
        for prediction_id in df_submission["prediction_id"]:
            parsed = parse_prediction_id(prediction_id)
            current_month = parsed["visit_month"]
            updrs_part = parsed["updrs_part"]
            target_col = f"updrs_{updrs_part}"

            if current_month in visit_preds:
                rating = visit_preds[current_month].get(target_col, np.nan)
            else:
                rating = np.nan
            ratings.append(rating)

        df_submission = df_submission.copy()
        df_submission["rating"] = ratings
        env.predict(df_submission[["prediction_id", "rating"]])

    return pd.read_csv(submission_path)


def score_submission(
    submission: pd.DataFrame,
    answer_key: Path = MOCK_DIR / "answer_key.csv",
) -> pd.DataFrame:
    """Score mock submission against local answer key (not available on real Kaggle test)."""
    answers = pd.read_csv(answer_key).rename(columns={"rating": "rating_true"})
    preds = submission.rename(columns={"rating": "rating_pred"})
    merged = preds.merge(answers, on="prediction_id")
    merged["updrs_part"] = merged["prediction_id"].map(
        lambda pid: parse_prediction_id(pid)["updrs_part"]
    )
    merged["target"] = merged["updrs_part"].map(lambda p: f"updrs_{p}")

    rows = []
    for target in UPDRS_TARGETS:
        sub = merged[merged["target"] == target]
        valid = sub["rating_true"].notna() & sub["rating_pred"].notna()
        if valid.sum() == 0:
            continue
        mae = np.mean(np.abs(sub.loc[valid, "rating_true"] - sub.loc[valid, "rating_pred"]))
        rows.append({"target": target, "mae": mae, "n": int(valid.sum())})
    return pd.DataFrame(rows)


def run_mock_api_evaluation(
    model: ProgressionModel | None = None,
    mock_dir: Path = MOCK_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(DATA_DIR / "train_clinical_data.csv")
    if model is None:
        model = ProgressionModel.load()

    submission = run_mock_api_inference(model, clinical, mock_dir=mock_dir)
    submission.to_csv(MODEL_DIR / "mock_submission.csv", index=False)
    scores = score_submission(submission, mock_dir / "answer_key.csv")
    scores.to_csv(MODEL_DIR / "mock_api_scores.csv", index=False)
    return submission, scores
