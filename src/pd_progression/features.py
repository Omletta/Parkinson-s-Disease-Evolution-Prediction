"""Feature engineering for UPDRS progression modeling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import CLINICAL_PATH, MEDICATION_COLUMN, PROTEINS_PATH, UPDRS_TARGETS


def _encode_medication(series: pd.Series) -> pd.DataFrame:
    med = series.fillna("Unknown")
    return pd.DataFrame(
        {
            "med_on": (med == "On").astype(float),
            "med_off": (med == "Off").astype(float),
            "med_unknown": (med == "Unknown").astype(float),
        },
        index=series.index,
    )


def _add_clinical_history(clinical: pd.DataFrame) -> pd.DataFrame:
    """Build temporal entry features without leaking current UPDRS into inputs."""
    df = clinical.sort_values(["patient_id", "visit_month"]).copy()
    grouped = df.groupby("patient_id", sort=False)

    df["months_since_prev_visit"] = grouped["visit_month"].diff()
    df["visit_index"] = grouped.cumcount()

    baseline = grouped[UPDRS_TARGETS].transform("first")
    df[[f"{col}_baseline" for col in UPDRS_TARGETS]] = baseline

    lagged = grouped[UPDRS_TARGETS].shift(1)
    lagged.columns = [f"{col}_lag1" for col in UPDRS_TARGETS]
    df[lagged.columns] = lagged

    for target in UPDRS_TARGETS:
        lag_col = f"{target}_lag1"
        base_col = f"{target}_baseline"
        df[f"{target}_change_from_baseline"] = df[lag_col] - df[base_col]
        df[f"{target}_months_progression"] = df["visit_month"].replace(0, np.nan)

    med = _encode_medication(df[MEDICATION_COLUMN])
    return pd.concat([df, med], axis=1)


def _proteins_wide(proteins: pd.DataFrame) -> pd.DataFrame:
    wide = proteins.pivot_table(
        index="visit_id",
        columns="UniProt",
        values="NPX",
        aggfunc="first",
    )
    wide.columns = [f"log_npx_{col}" for col in wide.columns]
    wide = np.log10(wide + 1.0)
    return wide


@dataclass
class FeatureBuilder:
    """Transform raw tables into model-ready entry features."""

    protein_columns: list[str] = field(default_factory=list)
    entry_columns: list[str] = field(default_factory=list)
    protein_fill_values: dict[str, float] = field(default_factory=dict)

    def fit(self, clinical: pd.DataFrame, proteins: pd.DataFrame) -> "FeatureBuilder":
        enriched = _add_clinical_history(clinical)
        protein_wide = _proteins_wide(proteins)

        self.protein_columns = protein_wide.columns.tolist()
        self.protein_fill_values = protein_wide.median(numeric_only=True).to_dict()

        static_cols = [
            "visit_month",
            "months_since_prev_visit",
            "visit_index",
            "med_on",
            "med_off",
            "med_unknown",
        ]
        history_cols = []
        for target in UPDRS_TARGETS:
            history_cols.extend(
                [
                    f"{target}_lag1",
                    f"{target}_baseline",
                    f"{target}_change_from_baseline",
                    f"{target}_months_progression",
                ]
            )
        self.entry_columns = static_cols + history_cols
        return self

    def transform(
        self,
        clinical: pd.DataFrame,
        proteins: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        enriched = _add_clinical_history(clinical)
        features = enriched[
            ["visit_id", "patient_id", "visit_month"] + self.entry_columns
        ].copy()

        if proteins is not None and len(proteins):
            protein_wide = _proteins_wide(proteins)
            protein_wide = protein_wide.reindex(columns=self.protein_columns)
            protein_wide = protein_wide.fillna(self.protein_fill_values)
            features = features.merge(
                protein_wide,
                left_on="visit_id",
                right_index=True,
                how="inner",
            )
        else:
            for col in self.protein_columns:
                features[col] = self.protein_fill_values.get(col, 0.0)

        targets = enriched[["visit_id"] + UPDRS_TARGETS].copy()
        if proteins is not None and len(proteins):
            visit_ids = set(features["visit_id"])
            targets = targets[targets["visit_id"].isin(visit_ids)]

        feature_cols = self.entry_columns + self.protein_columns
        x = features.set_index("visit_id")[feature_cols]
        y = targets.set_index("visit_id")[UPDRS_TARGETS]
        meta = features.set_index("visit_id")[["patient_id"]]
        return x.join(meta), y

    def fit_transform(
        self,
        clinical: pd.DataFrame,
        proteins: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.fit(clinical, proteins)
        return self.transform(clinical, proteins)


def load_training_tables(
    clinical_path=CLINICAL_PATH,
    proteins_path=PROTEINS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(clinical_path)
    proteins = pd.read_csv(proteins_path)
    return clinical, proteins
