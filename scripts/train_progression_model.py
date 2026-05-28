"""Train UPDRS progression model from engineered entry features."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pd_progression.constants import MODEL_DIR, UPDRS_TARGETS
from pd_progression.model import train_progression_pipeline


def main() -> None:
    model, cv_results, dataset = train_progression_pipeline(n_splits=5)
    summary = (
        cv_results.groupby("target")[["mae", "rmse"]]
        .mean()
        .round(3)
        .reset_index()
    )

    print("=== UPDRS progression model (5-fold patient CV) ===")
    print(summary.to_string(index=False))
    print(f"\nTraining rows (visits with CSF + clinical): {len(dataset)}")
    print(f"Feature count: {len(model.feature_builder.entry_columns + model.feature_builder.protein_columns)}")
    print(f"Saved model -> {MODEL_DIR / 'progression_model.joblib'}")

    cv_results.to_csv(MODEL_DIR / "cv_results.csv", index=False)
    summary.to_csv(MODEL_DIR / "cv_summary.csv", index=False)
    metrics = {
        "n_rows": len(dataset),
        "n_patients": int(dataset["patient_id"].nunique()),
        "n_features": len(model.feature_builder.entry_columns + model.feature_builder.protein_columns),
        "cv_summary": summary.set_index("target").to_dict(orient="index"),
    }
    (MODEL_DIR / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
