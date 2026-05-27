"""Generate EDA statistics and figures for AMP-PD peptide competition."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()


clinical = pd.read_csv(DATA / "train_clinical_data.csv")
supp = pd.read_csv(DATA / "supplemental_clinical_data.csv")
proteins = pd.read_csv(DATA / "train_proteins.csv")
peptides = pd.read_csv(DATA / "train_peptides.csv")

updrs_cols = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

# --- Figure 1: UPDRS distributions ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.ravel(), updrs_cols):
    sns.histplot(clinical[col].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title(f"{col} (train, n={clinical[col].notna().sum()})")
    ax.set_xlabel("Score")
savefig("01_updrs_distributions.png")

# --- Figure 2: UPDRS correlation ---
corr = clinical[updrs_cols].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1)
plt.title("UPDRS Part Correlations (train clinical)")
savefig("02_updrs_correlation.png")

# --- Figure 3: Disease progression over time ---
long_clin = clinical.melt(
    id_vars=["patient_id", "visit_month"],
    value_vars=updrs_cols,
    var_name="updrs_part",
    value_name="score",
)
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=long_clin,
    x="visit_month",
    y="score",
    hue="updrs_part",
    errorbar=("ci", 95),
)
plt.title("Mean UPDRS by Visit Month (train clinical)")
plt.xlabel("Months since baseline")
plt.ylabel("Mean score")
savefig("03_updrs_progression_over_time.png")

# --- Figure 4: Individual patient trajectories (sample) ---
sample_patients = clinical["patient_id"].drop_duplicates().sample(12, random_state=42)
traj = clinical[clinical["patient_id"].isin(sample_patients)]
fig, axes = plt.subplots(3, 4, figsize=(14, 10), sharex=True, sharey=False)
for ax, pid in zip(axes.ravel(), sample_patients):
    sub = traj[traj["patient_id"] == pid].sort_values("visit_month")
    for col in updrs_cols:
        ax.plot(sub["visit_month"], sub[col], marker="o", label=col, alpha=0.8)
    ax.set_title(f"Patient {pid}")
    ax.set_xlabel("Month")
    ax.set_ylabel("UPDRS")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
plt.suptitle("Sample Patient UPDRS Trajectories", y=1.05)
savefig("04_sample_patient_trajectories.png")

# --- Figure 5: Medication effect on UPDRS-3 ---
med = clinical.dropna(subset=["upd23b_clinical_state_on_medication", "updrs_3"])
plt.figure(figsize=(6, 5))
sns.boxplot(data=med, x="upd23b_clinical_state_on_medication", y="updrs_3")
plt.title("UPDRS Part 3 by Medication State")
savefig("05_medication_effect_updrs3.png")

# --- Figure 6: Visits per patient ---
vpc = clinical.groupby("patient_id").size()
plt.figure(figsize=(7, 4))
sns.histplot(vpc, bins=range(vpc.min(), vpc.max() + 2), kde=False)
plt.title("Visits per Patient (train clinical)")
plt.xlabel("Number of visits")
savefig("06_visits_per_patient.png")

# --- Figure 7: Clinical vs protein visit coverage ---
visit_flags = clinical[["visit_id", "patient_id", "visit_month"]].copy()
visit_flags["has_proteins"] = visit_flags["visit_id"].isin(proteins["visit_id"])
coverage = visit_flags.groupby("patient_id")["has_proteins"].mean()
plt.figure(figsize=(7, 4))
sns.histplot(coverage, bins=20, kde=False)
plt.title("Fraction of Clinical Visits with Protein Data (per patient)")
plt.xlabel("Protein coverage ratio")
savefig("07_protein_visit_coverage.png")

# --- Figure 8: Proteins measured per visit ---
prot_per_visit = proteins.groupby("visit_id").size()
plt.figure(figsize=(7, 4))
sns.histplot(prot_per_visit, bins=30, kde=False)
plt.title("Proteins Measured per Visit")
plt.xlabel("Count of protein records")
savefig("08_proteins_per_visit.png")

# --- Figure 9: NPX distribution (log scale) ---
plt.figure(figsize=(7, 4))
sns.histplot(np.log10(proteins["NPX"]), bins=50, kde=False)
plt.title("Log10(NPX) Distribution (train proteins)")
plt.xlabel("log10(NPX)")
savefig("09_npx_log_distribution.png")

# --- Figure 10: Protein-UPDRS correlation (top features) ---
merged = proteins.merge(
    clinical[["visit_id"] + updrs_cols],
    on="visit_id",
    how="inner",
)
pivot = merged.pivot_table(index="visit_id", columns="UniProt", values="NPX", aggfunc="first")
targets = merged.drop_duplicates("visit_id").set_index("visit_id")[updrs_cols]

corrs = {}
for target in updrs_cols:
    y = targets[target]
    valid = y.notna()
    c = pivot.loc[valid].apply(lambda s: s.corr(y.loc[valid]), axis=0)
    corrs[target] = c

corr_df = pd.DataFrame(corrs)
top = corr_df.abs().max(axis=1).sort_values(ascending=False).head(20).index
top_corr = corr_df.loc[top]

plt.figure(figsize=(10, 8))
sns.heatmap(top_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Top 20 Proteins by |correlation| with any UPDRS part")
plt.ylabel("UniProt")
savefig("10_top_protein_updrs_correlations.png")

# --- Figure 11: Supplemental vs train progression ---
supp_long = supp.melt(
    id_vars=["visit_month"],
    value_vars=updrs_cols,
    var_name="updrs_part",
    value_name="score",
)
train_long = clinical.melt(
    id_vars=["visit_month"],
    value_vars=updrs_cols,
    var_name="updrs_part",
    value_name="score",
)
train_long["cohort"] = "train (with CSF)"
supp_long["cohort"] = "supplemental (clinical only)"

combined = pd.concat([train_long, supp_long], ignore_index=True)
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=combined,
    x="visit_month",
    y="score",
    hue="updrs_part",
    style="cohort",
    errorbar=("ci", 95),
)
plt.title("UPDRS Progression: Train vs Supplemental Cohorts")
plt.xlabel("Months since baseline")
savefig("11_train_vs_supplemental_progression.png")

# --- Summary stats to markdown ---
summary_lines = [
    "# EDA Summary Statistics",
    "",
    "## Dataset sizes",
    f"- train_clinical_data: {clinical.shape[0]} rows, {clinical.patient_id.nunique()} patients",
    f"- supplemental_clinical_data: {supp.shape[0]} rows, {supp.patient_id.nunique()} patients",
    f"- train_proteins: {proteins.shape[0]} rows, {proteins.UniProt.nunique()} unique proteins",
    f"- train_peptides: {peptides.shape[0]} rows, {peptides.Peptide.nunique()} unique peptides",
    "",
    "## Target availability (train clinical)",
]
for col in updrs_cols:
    n = clinical[col].notna().sum()
    summary_lines.append(f"- {col}: {n} non-null ({100*n/len(clinical):.1f}%)")

summary_lines.extend([
    "",
    "## Join coverage",
    f"- Clinical visits: {clinical.visit_id.nunique()}",
    f"- Visits with protein data: {visit_flags['has_proteins'].sum()} ({100*visit_flags['has_proteins'].mean():.1f}%)",
    f"- Visits without protein data: {(~visit_flags['has_proteins']).sum()}",
    "",
    "## Medication flag",
    f"- On medication assessments: {(clinical['upd23b_clinical_state_on_medication']=='On').sum()}",
    f"- Off medication assessments: {(clinical['upd23b_clinical_state_on_medication']=='Off').sum()}",
    f"- Missing medication flag: {clinical['upd23b_clinical_state_on_medication'].isna().sum()}",
    "",
    "## Modeling notes",
    "- UPDRS parts are correlated; joint or multi-target models may help.",
    "- Part 4 has heavy missingness (~60%); treat as auxiliary or impute carefully.",
    "- Only ~41% of clinical visits have matched CSF protein measurements.",
    "- Supplemental cohort provides progression context without omics (771 patients, no overlap with train).",
    "- Medication state strongly affects UPDRS-3; include as feature or stratify.",
    "",
    "## Top protein correlations (absolute, any UPDRS part)",
])
for prot in top[:10]:
    row = corr_df.loc[prot]
    best_part = row.abs().idxmax()
    summary_lines.append(f"- {prot}: r={row[best_part]:.3f} with {best_part}")

(OUT / "eda_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
print(f"Wrote figures and summary to {OUT}")
