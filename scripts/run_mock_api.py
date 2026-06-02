"""Run offline mock time-series API inference and score vs answer_key."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pd_progression.constants import MOCK_DIR, MODEL_DIR
from pd_progression.mock_api import run_mock_api_evaluation


def main() -> None:
    if not (MOCK_DIR / "sample_submission.csv").exists():
        print("Mock API files not found. Run: python scripts/build_mock_api_data.py")
        sys.exit(1)
    if not (MODEL_DIR / "progression_model.joblib").exists():
        print("Model not found. Run: python scripts/train_progression_model.py --mode mock")
        sys.exit(1)

    submission, scores = run_mock_api_evaluation()
    print("=== Mock API evaluation (held-out patients via API batches) ===")
    print(scores.to_string(index=False))
    print(f"\nSubmission -> {MODEL_DIR / 'mock_submission.csv'}")
    print(f"Scores     -> {MODEL_DIR / 'mock_api_scores.csv'}")


if __name__ == "__main__":
    main()
