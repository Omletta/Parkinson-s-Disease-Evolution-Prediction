"""Multi-head UPDRS progression model with masked per-target training."""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from .constants import MODEL_DIR, UPDRS_TARGETS
from .features import FeatureBuilder, load_training_tables
from .metrics import masked_mae, masked_rmse
from .split import PatientSplit, filter_by_patients, split_patients_with_proteins


@dataclass
class ProgressionModel:
    """Separate regression head per UPDRS part; each head ignores NaN targets."""

    feature_builder: FeatureBuilder = field(default_factory=FeatureBuilder)
    heads: dict[str, HistGradientBoostingRegressor] = field(default_factory=dict)
    random_state: int = 42

    def _default_head(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=self.random_state,
        )

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> "ProgressionModel":
        feature_cols = self.feature_builder.entry_columns + self.feature_builder.protein_columns
        x_matrix = x[feature_cols].to_numpy(dtype=float)

        self.heads = {}
        for target in UPDRS_TARGETS:
            mask = y[target].notna().to_numpy()
            head = self._default_head()
            head.fit(x_matrix[mask], y.loc[mask, target].to_numpy(dtype=float))
            self.heads[target] = head
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        feature_cols = self.feature_builder.entry_columns + self.feature_builder.protein_columns
        x_matrix = x[feature_cols].to_numpy(dtype=float)
        preds = {target: self.heads[target].predict(x_matrix) for target in UPDRS_TARGETS}
        return pd.DataFrame(preds, index=x.index)

    def cross_validate(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        groups: pd.Series,
        n_splits: int = 5,
    ) -> pd.DataFrame:
        gkf = GroupKFold(n_splits=n_splits)
        fold_rows = []

        feature_cols = self.feature_builder.entry_columns + self.feature_builder.protein_columns
        x_features = x[feature_cols]
        group_values = groups.loc[x.index].to_numpy()

        for fold, (train_idx, valid_idx) in enumerate(
            gkf.split(x_features, y, group_values),
            start=1,
        ):
            model = ProgressionModel(
                feature_builder=self.feature_builder,
                random_state=self.random_state,
            )
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]

            model.fit(x.iloc[train_idx], y_train)
            preds = model.predict(x.iloc[valid_idx])

            mae = masked_mae(y_valid, preds)
            rmse = masked_rmse(y_valid, preds)
            for target in UPDRS_TARGETS:
                fold_rows.append(
                    {
                        "fold": fold,
                        "target": target,
                        "mae": mae[target],
                        "rmse": rmse[target],
                        "n_valid": y_valid[target].notna().sum(),
                    }
                )

        return pd.DataFrame(fold_rows)

    def save(self, path=None) -> None:
        path = path or MODEL_DIR / "progression_model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=None) -> "ProgressionModel":
        path = path or MODEL_DIR / "progression_model.joblib"
        return joblib.load(path)


def evaluate_split(
    model: ProgressionModel,
    x: pd.DataFrame,
    y: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    preds = model.predict(x)
    mae = masked_mae(y, preds)
    rmse = masked_rmse(y, preds)
    rows = []
    for target in UPDRS_TARGETS:
        rows.append(
            {
                "split": split_name,
                "target": target,
                "mae": mae[target],
                "rmse": rmse[target],
                "n_rows": y[target].notna().sum(),
            }
        )
    return pd.DataFrame(rows)


def train_progression_pipeline(
    n_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    use_holdout: bool = True,
    exclude_patients: set[int] | None = None,
) -> dict:
    """
    Train the progression model.

    use_holdout=True: patient-level train/test split (default).
    use_holdout=False: train on all CSF visits (for mock API workflow).
    exclude_patients: optional set of patient IDs to drop from training
        (e.g. patients reserved for mock_api test files).
    """
    clinical, proteins = load_training_tables()
    patient_split = None
    proteins_train = proteins
    proteins_test = None

    if use_holdout:
        patient_split = split_patients_with_proteins(
            proteins,
            test_size=test_size,
            random_state=random_state,
        )
        patient_split.save()
        proteins_train = filter_by_patients(proteins, patient_split.train_patient_set)
        proteins_test = filter_by_patients(proteins, patient_split.test_patient_set)

    if exclude_patients:
        proteins_train = filter_by_patients(
            proteins_train,
            set(proteins_train["patient_id"].unique()) - exclude_patients,
        )

    builder = FeatureBuilder()
    builder.fit(clinical, proteins_train)
    x_train, y_train = builder.transform(clinical, proteins_train)

    model = ProgressionModel(feature_builder=builder, random_state=random_state)
    cv_results = model.cross_validate(
        x_train,
        y_train,
        groups=x_train["patient_id"],
        n_splits=n_splits,
    )
    model.fit(x_train, y_train)
    model.save()
    joblib.dump(builder, MODEL_DIR / "feature_builder.joblib")

    result = {
        "model": model,
        "patient_split": patient_split,
        "cv_results": cv_results,
        "x_train": x_train,
        "y_train": y_train,
    }

    if use_holdout and proteins_test is not None:
        x_test, y_test = builder.transform(clinical, proteins_test)
        test_results = evaluate_split(model, x_test, y_test, split_name="test")
        test_results.to_csv(MODEL_DIR / "test_results.csv", index=False)
        model.predict(x_test).join(y_test, rsuffix="_true").join(
            x_test[["patient_id"]]
        ).to_csv(MODEL_DIR / "test_predictions.csv")
        result.update(
            {
                "test_results": test_results,
                "train_eval": evaluate_split(model, x_train, y_train, "train"),
                "x_test": x_test,
                "y_test": y_test,
            }
        )

    return result
