import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
import joblib

FEATURE_COLS = ["mean_rr_interval", "heart_rate_bpm", "p_wave_amplitude",
                 "qrs_amplitude", "t_wave_amplitude", "pr_interval",
                 "qt_interval", "qrs_duration", "n_beats_detected"]
FINAL_CLASSES = ["AF", "IAVB", "LAD", "LBBB", "NSIVCB", "NSR", "PAC", "QAb", "RBBB", "SB", "STach", "TAb"]

fiducial_df = pd.read_csv(r"C:\ECG_Project\training\ecg_fiducial_features_clean.csv")

train_df = fiducial_df[fiducial_df["split"] == "train"]
test_df = fiducial_df[fiducial_df["split"] == "test"]

X_train, y_train = train_df[FEATURE_COLS], train_df[FINAL_CLASSES]
X_test, y_test = test_df[FEATURE_COLS], test_df[FINAL_CLASSES]

# One RandomForest per class (multi-label), same 12-class structure as your ResNet

base_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
clf = MultiOutputClassifier(base_rf)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
print(f"Fiducial-feature classifier — micro_f1={micro_f1:.4f}  macro_f1={macro_f1:.4f}")

joblib.dump(clf, r"C:\ECG_Project\training\fiducial_rf_classifier.pkl")
print("Saved: fiducial_rf_classifier.pkl")
