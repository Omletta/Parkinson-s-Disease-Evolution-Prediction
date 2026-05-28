from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "outputs" / "models"

UPDRS_TARGETS = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]
MEDICATION_COLUMN = "upd23b_clinical_state_on_medication"

CLINICAL_PATH = DATA_DIR / "train_clinical_data.csv"
SUPPLEMENTAL_PATH = DATA_DIR / "supplemental_clinical_data.csv"
PROTEINS_PATH = DATA_DIR / "train_proteins.csv"
PEPTIDES_PATH = DATA_DIR / "train_peptides.csv"
