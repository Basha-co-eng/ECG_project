# ============ START: label_harmonization.py ============
import pandas as pd
import numpy as np
import requests


def load_snomed_mapping():
    """Official SNOMED -> diagnosis label mapping from the PhysioNet Challenge evaluation repo."""
    mapping_url = "https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/main/dx_mapping_scored.csv"
    dx_mapping = pd.read_csv(mapping_url)
    return dx_mapping


# Final shared taxonomy: SNOMED code -> class name
# Includes the CLBBB -> LBBB merge (733534002 scored same as 164909002)
CLASS_MAP = {
    "426783006": "NSR",     # Normal sinus rhythm
    "164889003": "AF",      # Atrial fibrillation
    "270492004": "IAVB",    # 1st degree AV block
    "39732003":  "LAD",     # Left axis deviation
    "164909002": "LBBB",    # Left bundle branch block
    "733534002": "LBBB",    # Complete LBBB -> merged into LBBB
    "698252002": "NSIVCB",  # Nonspecific intraventricular conduction disorder
    "284470004": "PAC",     # Premature atrial contraction
    "164917005": "QAb",     # Q wave abnormal
    "59118001":  "RBBB",    # Right bundle branch block
    "426177001": "SB",      # Sinus bradycardia
    "427084000": "STach",   # Sinus tachycardia
    "164934002": "TAb",     # T wave abnormal
}

FINAL_CLASSES = sorted(set(CLASS_MAP.values()))


def encode_labels(dx_codes_str: str) -> np.ndarray:
    """
    Takes the raw 'dx_codes' string (e.g. '426177001,55827005,164934002')
    and returns a multi-hot vector over FINAL_CLASSES.
    """
    label_vec = np.zeros(len(FINAL_CLASSES), dtype=np.int8)
    if pd.isna(dx_codes_str):
        return label_vec

    codes = [c.strip() for c in str(dx_codes_str).split(",")]
    for code in codes:
        if code in CLASS_MAP:
            class_name = CLASS_MAP[code]
            idx = FINAL_CLASSES.index(class_name)
            label_vec[idx] = 1
    return label_vec


def harmonize_labels(metadata_df: pd.DataFrame) -> pd.DataFrame:
    print(f"Final taxonomy ({len(FINAL_CLASSES)} classes):", FINAL_CLASSES)

    label_matrix = np.stack(metadata_df["dx_codes"].apply(encode_labels).values)
    label_df = pd.DataFrame(label_matrix, columns=FINAL_CLASSES)

    metadata_df = pd.concat([metadata_df.reset_index(drop=True), label_df], axis=1)

    metadata_df["has_label"] = metadata_df[FINAL_CLASSES].sum(axis=1) > 0
    print("Records with at least one relevant label:", metadata_df["has_label"].sum())
    print("Records dropped (no relevant label):", (~metadata_df["has_label"]).sum())

    metadata_df = metadata_df[metadata_df["has_label"]].reset_index(drop=True)

    coverage = metadata_df.groupby("source_hospital")[FINAL_CLASSES].sum()
    print(coverage)

    return metadata_df


if __name__ == "__main__":
    metadata_df = pd.read_csv(r"C:\ECG_Project\training\ecg_metadata_inventory.csv")
    print(metadata_df.shape)

    metadata_df = harmonize_labels(metadata_df)

    out_path = r"C:\ECG_Project\training\ecg_metadata_cleaned_labeled.csv"
    metadata_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
# ============ END: label_harmonization.py ============
