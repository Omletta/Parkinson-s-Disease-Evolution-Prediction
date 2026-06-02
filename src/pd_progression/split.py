"""Patient-level train/test splitting (no visit leakage across sets)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import MODEL_DIR


@dataclass
class PatientSplit:
    train_patients: list[int]
    test_patients: list[int]
    test_size: float
    random_state: int

    @property
    def train_patient_set(self) -> set[int]:
        return set(self.train_patients)

    @property
    def test_patient_set(self) -> set[int]:
        return set(self.test_patients)

    def save(self, path=None) -> None:
        path = path or MODEL_DIR / "patient_split.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "train_patients": self.train_patients,
            "test_patients": self.test_patients,
            "test_size": self.test_size,
            "random_state": self.random_state,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load(path=None) -> "PatientSplit":
        path = path or MODEL_DIR / "patient_split.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return PatientSplit(
            train_patients=payload["train_patients"],
            test_patients=payload["test_patients"],
            test_size=payload["test_size"],
            random_state=payload["random_state"],
        )


def split_patients_with_proteins(
    proteins: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PatientSplit:
    """Hold out whole patients; all visits from a patient stay in one split."""
    patients = np.array(sorted(proteins["patient_id"].unique()))
    rng = np.random.RandomState(random_state)
    shuffled = rng.permutation(patients)

    n_test = max(1, int(round(len(patients) * test_size)))
    test_patients = sorted(shuffled[:n_test].tolist())
    train_patients = sorted(shuffled[n_test:].tolist())

    return PatientSplit(
        train_patients=train_patients,
        test_patients=test_patients,
        test_size=test_size,
        random_state=random_state,
    )


def filter_by_patients(df: pd.DataFrame, patients: set[int]) -> pd.DataFrame:
    return df[df["patient_id"].isin(patients)].copy()
