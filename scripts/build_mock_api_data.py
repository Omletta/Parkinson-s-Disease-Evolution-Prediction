"""Create local mock API CSVs from train data (mimics Kaggle example_test_files)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pd_progression.constants import MOCK_DIR
from pd_progression.mock_api import build_mock_api_files


def main() -> None:
    split = build_mock_api_files(test_size=0.2, random_state=42)
    print(f"Wrote mock API files to {MOCK_DIR}")
    print(f"  Train patients (for --mock mode): {len(split.train_patients)}")
    print(f"  Mock test patients (API only):    {len(split.test_patients)}")
    print("\nNext:")
    print("  python scripts/train_progression_model.py --mode mock")
    print("  python scripts/run_mock_api.py")


if __name__ == "__main__":
    main()
