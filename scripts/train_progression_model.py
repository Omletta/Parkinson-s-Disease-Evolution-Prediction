"""Train UPDRS progression model from engineered entry features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pd_progression.constants import MOCK_DIR, MODEL_DIR
from pd_progression.mock_api import build_mock_api_files, run_mock_api_evaluation
from pd_progression.model import train_progression_pipeline
from pd_progression.split import PatientSplit


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UPDRS progression model")
    parser.add_argument(
        "--mode",
        choices=["holdout", "full", "mock"],
        default="holdout",
        help=(
            "holdout: patient train/test split; "
            "full: train on all CSF visits; "
            "mock: train excluding mock_api test patients, then run mock API"
        ),
    )
    args = parser.parse_args()

    use_holdout = args.mode == "holdout"
    exclude_patients = None

    if args.mode == "mock":
        if not (MOCK_DIR / "patient_split.json").exists():
            build_mock_api_files()
        split = PatientSplit.load(MOCK_DIR / "patient_split.json")
        exclude_patients = split.test_patient_set
        use_holdout = False

    result = train_progression_pipeline(
        n_splits=5,
        test_size=0.2,
        random_state=42,
        use_holdout=use_holdout,
        exclude_patients=exclude_patients,
    )
    model = result["model"]
    cv_results = result["cv_results"]
    cv_summary = (
        cv_results.groupby("target")[["mae", "rmse"]]
        .mean()
        .round(3)
        .reset_index()
    )

    print(f"=== Training mode: {args.mode} ===")
    print(cv_summary.to_string(index=False))
    print(f"\nTrain visits: {len(result['x_train'])}")
    print(f"Saved model -> {MODEL_DIR / 'progression_model.joblib'}")

    cv_results.to_csv(MODEL_DIR / "cv_results.csv", index=False)
    cv_summary.to_csv(MODEL_DIR / "cv_summary.csv", index=False)

    metrics = {
        "mode": args.mode,
        "n_train_rows": len(result["x_train"]),
        "n_train_patients": int(result["x_train"]["patient_id"].nunique()),
        "n_features": len(
            model.feature_builder.entry_columns + model.feature_builder.protein_columns
        ),
        "cv_summary": cv_summary.set_index("target").to_dict(orient="index"),
    }

    if args.mode == "holdout" and "test_results" in result:
        test_summary = result["test_results"].set_index("target")[["mae", "rmse"]]
        print("\n=== Held-out test patients ===")
        print(test_summary.round(3).to_string())
        metrics["test_summary"] = test_summary.round(3).to_dict(orient="index")

    if args.mode == "mock":
        print("\n=== Running mock time-series API ===")
        _, mock_scores = run_mock_api_evaluation(model)
        print(mock_scores.to_string(index=False))
        metrics["mock_api_summary"] = mock_scores.set_index("target").to_dict(orient="index")

    (MODEL_DIR / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
