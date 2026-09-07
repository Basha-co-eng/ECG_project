import pandas as pd
import numpy as np

fiducial_df = pd.read_csv(r"C:\ECG_Project\training\ecg_fiducial_features.csv")
metadata_df = pd.read_csv(r"C:\ECG_Project\training\ecg_metadata_train_test_split.csv")

FINAL_CLASSES = ["AF", "IAVB", "LAD", "LBBB", "NSIVCB", "NSR", "PAC", "QAb", "RBBB", "SB", "STach", "TAb"]

# Bring in source_hospital + split, so imputation can be done per-hospital
# and we keep the same train/test assignment as your ResNet pipeline
fiducial_df = fiducial_df.merge(
    metadata_df[["record_id", "source_hospital", "split"] + FINAL_CLASSES],
    on="record_id", how="inner"
)
print(f"After merge: {fiducial_df.shape}")

FEATURE_COLS = ["mean_rr_interval", "heart_rate_bpm", "p_wave_amplitude",
                 "qrs_amplitude", "t_wave_amplitude", "pr_interval",
                 "qt_interval", "qrs_duration", "n_beats_detected"]

print("\nNull counts before imputation:")
print(fiducial_df[FEATURE_COLS].isnull().sum())

# Per-hospital median imputation — fit only on TRAIN rows, applied to both
# train and test (avoids leaking test-set statistics into training, same
# discipline as your normalization/scaling steps elsewhere in the pipeline)
for col in FEATURE_COLS:
    train_medians = fiducial_df[fiducial_df["split"] == "train"].groupby("source_hospital")[col].median()

    def fill_value(row):
        if pd.isna(row[col]):
            return train_medians.get(row["source_hospital"], fiducial_df[col].median())
        return row[col]

    fiducial_df[col] = fiducial_df.apply(fill_value, axis=1)

print("\nNull counts after imputation:")
print(fiducial_df[FEATURE_COLS].isnull().sum())   # should all be 0 now

fiducial_df.to_csv(r"C:\ECG_Project\training\ecg_fiducial_features_clean.csv", index=False)
print(f"\nSaved: ecg_fiducial_features_clean.csv, shape={fiducial_df.shape}")
