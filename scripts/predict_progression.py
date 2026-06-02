"""Evaluate the saved model on the held-out test patients only."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pd_progression.constants import CLINICAL_PATH, MODEL_DIR, PROTEINS_PATH, UPDRS_TARGETS
from pd_progression.metrics import masked_mae, masked_rmse
from pd_progression.model import ProgressionModel
from pd_progression.split import PatientSplit, filter_by_patients


def main() -> None:
    clinical = pd.read_csv(CLINICAL_PATH)
    proteins = pd.read_csv(PROTEINS_PATH)
    patient_split = PatientSplit.load()
    model = ProgressionModel.load()

    proteins_test = filter_by_patients(proteins, patient_split.test_patient_set)
    x_test, y_test = model.feature_builder.transform(clinical, proteins_test)
    preds = model.predict(x_test)

    out = preds.join(y_test, rsuffix="_true").join(x_test[["patient_id"]])
    out.to_csv(MODEL_DIR / "test_predictions.csv")

    mae = masked_mae(y_test, preds)
    rmse = masked_rmse(y_test, preds)
    print("=== Held-out test set (unseen patients) ===")
    print(f"Test patients: {len(patient_split.test_patients)}")
    print(f"Test visits:   {len(x_test)}\n")
    for target in UPDRS_TARGETS:
        print(f"  {target}: MAE={mae[target]:.3f}, RMSE={rmse[target]:.3f}")
    print(f"\nWrote {MODEL_DIR / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
