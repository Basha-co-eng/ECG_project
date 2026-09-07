import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from itertools import combinations

def is_degenerate(heatmap):
    return (heatmap.max() - heatmap.min()) < 1e-6

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

gradcam_data = np.load(r"C:\ECG_Project\training\stage10_gradcam\gradcam_heatmaps.npz")
FINAL_CLASSES = ["AF", "IAVB", "LAD", "LBBB", "NSIVCB", "NSR", "PAC", "QAb", "RBBB", "SB", "STach", "TAb"]
FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]
shap_consistency = {}
gradcam_consistency = {}

uncertainty_agg = {}
degenerate_cells = []

for model_name in ["fedavg", "fedprox", "fedopt"]:
    for cls in FINAL_CLASSES:
        available = {}
        for h in FEDERATED_CLIENTS:
            key = f"{model_name}__{h}__{cls}"
            if key in gradcam_data:
                available[h] = gradcam_data[key]

        if len(available) < 2:
            continue

        if all(is_degenerate(v) for v in available.values()):
            degenerate_cells.append(f"{model_name}__{cls}")
            continue

        pairwise = [cosine_sim(a, b) for (_, a), (_, b) in combinations(available.items(), 2)]
        gradcam_consistency[f"{model_name}__{cls}"] = np.mean(pairwise)

print(f"Degenerate cells excluded ({len(degenerate_cells)}): {degenerate_cells}")

# Rebuild phase_d_df, now over 3 models, skipping any (model, class) pair with no
# valid gradcam_consistency entry (i.e. the degenerate ones)
rows = []
for model_name in ["fedavg", "fedprox", "fedopt"]:
    for cls in FINAL_CLASSES:
        gc_key = f"{model_name}__{cls}"
        unc_key = f"{model_name}__{cls}"

        if gc_key not in gradcam_consistency or unc_key not in uncertainty_agg or cls not in shap_consistency:
            continue

        rows.append({
            "model": model_name,
            "class": cls,
            "gradcam_consistency": gradcam_consistency[gc_key],
            "shap_consistency": shap_consistency[cls],
            "uncertainty_level": uncertainty_agg[unc_key]["uncertainty_level"],
            "uncertainty_spread": uncertainty_agg[unc_key]["uncertainty_spread"],
        })

phase_d_clean = pd.DataFrame(rows)
print(f"\nClean rows (degenerate cells excluded): {len(phase_d_clean)} of 36 possible")
print(phase_d_clean)

phase_d_clean.to_csv(r"C:\ECG_Project\training\phase_d_consistency_uncertainty_CLEAN.csv", index=False)

# Re-run correlations on the CLEAN data only
inconsistency = 1 - phase_d_clean["gradcam_consistency"]
shap_inconsistency = 1 - phase_d_clean["shap_consistency"]

print("\n=== CLEAN Pearson ===")
for label, col in [("Grad-CAM", inconsistency), ("SHAP", shap_inconsistency)]:
    for unc_col in ["uncertainty_level", "uncertainty_spread"]:
        r, p = pearsonr(col, phase_d_clean[unc_col])
        print(f"  {label} vs {unc_col}: r={r:.4f}, p={p:.4f}")

print("\n=== CLEAN Spearman ===")
for label, col in [("Grad-CAM", inconsistency), ("SHAP", shap_inconsistency)]:
    for unc_col in ["uncertainty_level", "uncertainty_spread"]:
        r, p = spearmanr(col, phase_d_clean[unc_col])
        print(f"  {label} vs {unc_col}: r={r:.4f}, p={p:.4f}")