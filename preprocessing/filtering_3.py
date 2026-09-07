# ============ START: quality_filter.py ============
import pandas as pd

# 5 real federated clients — ptb and st_petersburg_incart are now DROPPED ENTIRELY,
# not held out. (Earlier version kept them as "external_holdout"; per your call,
# 5 real hospitals already exceeds FedCVD's 4, so no holdout set is kept.)
FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]


def filter_to_five_clients(metadata_df: pd.DataFrame) -> pd.DataFrame:
    # Merge CPSC main + extra into a single "cpsc_2018" client (same institution/device family)
    metadata_df["source_hospital"] = metadata_df["source_hospital"].replace(
        {"cpsc_2018_extra": "cpsc_2018"}
    )

    before = len(metadata_df)
    metadata_df = metadata_df[metadata_df["source_hospital"].isin(FEDERATED_CLIENTS)].reset_index(drop=True)
    print(f"Dropped {before - len(metadata_df)} records from ptb/st_petersburg_incart (no longer used)")
    print(metadata_df["source_hospital"].value_counts())

    return metadata_df


def quality_report(metadata_df: pd.DataFrame):
    print("Sampling rate distribution:")
    print(metadata_df["sampling_rate"].value_counts())

    print("\nLead count distribution (should be 12 everywhere):")
    print(metadata_df["num_leads"].value_counts())

    seconds = metadata_df["num_samples"] / metadata_df["sampling_rate"]
    metadata_df["duration_sec"] = seconds
    print("\nSignal length (seconds) distribution:")
    print(seconds.describe())

    print("\nSampling rate by source hospital:")
    print(pd.crosstab(metadata_df["source_hospital"], metadata_df["sampling_rate"]))

    print("\nDuration stats by source hospital:")
    print(metadata_df.groupby("source_hospital")["duration_sec"].describe())

    print("\nTop 10 longest recordings:")
    print(metadata_df.nlargest(10, "duration_sec")[
        ["record_id", "source_hospital", "duration_sec", "sampling_rate"]
    ])

    return metadata_df


if __name__ == "__main__":
    metadata_df = pd.read_csv(r"C:\ECG_Project\training\ecg_metadata_cleaned_labeled.csv")

    metadata_df = filter_to_five_clients(metadata_df)
    metadata_df = quality_report(metadata_df)

    out_path = r"C:\ECG_Project\training\ecg_metadata_final_5clients.csv"
    metadata_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"\nFinal shape: {metadata_df.shape}")
    print(f"Federated clients: {FEDERATED_CLIENTS}")
# ============ END: quality_filter.py ============
