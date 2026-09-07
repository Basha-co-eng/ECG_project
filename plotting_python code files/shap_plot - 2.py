import shap
import pandas as pd
import matplotlib.pyplot as plt
import joblib
FIGDIR = r"C:\ECG_Project\training\figures"
FINAL_CLASSES = ["AF", "IAVB", "LAD", "LBBB", "NSIVCB", "NSR", "PAC", "QAb", "RBBB", "SB", "STach", "TAb"]
clf = joblib.load(r"C:\ECG_Project\training\fiducial_rf_classifier.pkl")
FEATURE_COLS = ["mean_rr_interval", "heart_rate_bpm", "p_wave_amplitude",
                 "qrs_amplitude", "t_wave_amplitude", "pr_interval",
                 "qt_interval", "qrs_duration", "n_beats_detected"]
class_name = "AF"   # swap to inspect other diseases
class_idx = FINAL_CLASSES.index(class_name)
estimator = clf.estimators_[class_idx]
explainer = shap.TreeExplainer(estimator)

fiducial_df = pd.read_csv(r"C:\ECG_Project\training\All excel files\ecg_fiducial_features_clean.csv")
test_df = fiducial_df[fiducial_df["split"] == "test"]
pos_records = test_df[test_df[class_name] == 1]

X_subset = pos_records[FEATURE_COLS]
shap_values = explainer.shap_values(X_subset)
if isinstance(shap_values, list):
    sv = shap_values[1]
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

shap.summary_plot(sv, X_subset, feature_names=FEATURE_COLS, show=False)
plt.title(f"SHAP Summary — {class_name} (all hospitals pooled)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(rf"{FIGDIR}\shap_beeswarm_{class_name}.png", bbox_inches="tight")
plt.show()
