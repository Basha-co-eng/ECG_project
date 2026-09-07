import pandas as pd
import numpy as np
import shap
import joblib

FEATURE_COLS = ["mean_rr_interval", "heart_rate_bpm", "p_wave_amplitude",
                 "qrs_amplitude", "t_wave_amplitude", "pr_interval",
                 "qt_interval", "qrs_duration", "n_beats_detected"]
FINAL_CLASSES = ["AF", "IAVB", "LAD", "LBBB", "NSIVCB", "NSR", "PAC", "QAb", "RBBB", "SB", "STach", "TAb"]
FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]

# Load the already-trained classifier from disk — no need for base_rf/clf to
# already be in memory, since this may run in a different session/file
clf = joblib.load(r"C:\ECG_Project\training\fiducial_rf_classifier.pkl")

fiducial_df = pd.read_csv(r"C:\ECG_Project\training\ecg_fiducial_features_clean.csv")
test_df = fiducial_df[fiducial_df["split"] == "test"]

shap_results = {}

for class_idx, class_name in enumerate(FINAL_CLASSES):
    estimator = clf.estimators_[class_idx]   # the actual TRAINED model for this class
    class_explainer = shap.TreeExplainer(estimator)

    for hospital in FEDERATED_CLIENTS:
        hosp_test = test_df[test_df["source_hospital"] == hospital]
        pos_records = hosp_test[hosp_test[class_name] == 1]

        if len(pos_records) < 5:
            print(f"  [{hospital}/{class_name}] n={len(pos_records)}, below min — skipped")
            continue

        X_subset = pos_records[FEATURE_COLS]
        shap_values = class_explainer.shap_values(X_subset)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]   # positive class

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_results[f"{hospital}__{class_name}"] = dict(zip(FEATURE_COLS, mean_abs_shap))
        print(f"  [{hospital}/{class_name}] n={len(pos_records)} — SHAP computed")

shap_summary_df = pd.DataFrame(shap_results).T
shap_summary_df.to_csv(r"C:\ECG_Project\training\shap_feature_importance_summary.csv")
print(f"\nSaved: shap_feature_importance_summary.csv, shape={shap_summary_df.shape}")
