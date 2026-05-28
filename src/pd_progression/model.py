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


def train_progression_pipeline(
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[ProgressionModel, pd.DataFrame, pd.DataFrame]:
    clinical, proteins = load_training_tables()
    builder = FeatureBuilder()
    x, y = builder.fit_transform(clinical, proteins)

    model = ProgressionModel(feature_builder=builder, random_state=random_state)
    cv_results = model.cross_validate(
        x,
        y,
        groups=x["patient_id"],
        n_splits=n_splits,
    )
    model.fit(x, y)
    model.save()
    builder_path = MODEL_DIR / "feature_builder.joblib"
    joblib.dump(builder, builder_path)
    return model, cv_results, x.join(y)
