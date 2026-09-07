# ============ START: train_test_split.py ============
import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]

FINAL_CLASSES = ["AF", "IAVB", "LAD", "LBBB", "NSIVCB", "NSR", "PAC", "QAb", "RBBB", "SB", "STach", "TAb"]

STAGE4_INPUT_PATH = r"C:\ECG_Project\training\ecg_metadata_stage4_final.csv"
STAGE6_OUTPUT_PATH = r"C:\ECG_Project\training\ecg_metadata_train_test_split.csv"


def split_client(client_df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
    """
    Multi-label stratified split for ONE hospital's records, so rare classes
    stay represented in both train and test (a plain random split can wipe
    a rare class out of one side entirely).
    """
    client_df = client_df.reset_index(drop=True)

    X = np.arange(len(client_df)).reshape(-1, 1)  # dummy index array, not real features
    y = client_df[FINAL_CLASSES].values

    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=test_size)

    train_indices = X_train.flatten()
    test_indices = X_test.flatten()

    client_df["split"] = "train"
    client_df.loc[test_indices, "split"] = "test"

    return client_df


def split_all_clients(metadata_df: pd.DataFrame) -> pd.DataFrame:
    split_records = []

    for hospital in FEDERATED_CLIENTS:
        client_df = metadata_df[metadata_df["source_hospital"] == hospital]
        if len(client_df) == 0:
            print(f"WARNING: no records found for {hospital} — check source_hospital naming")
            continue

        client_df = split_client(client_df)

        n_train = (client_df["split"] == "train").sum()
        n_test = (client_df["split"] == "test").sum()
        print(f"{hospital}: {n_train} train / {n_test} test")

        split_records.append(client_df)

    final_df = pd.concat(split_records, ignore_index=True)
    return final_df


def verify_class_balance(final_df: pd.DataFrame):
    """Sanity check: confirm no class got completely wiped out of train or test for any hospital."""
    for hospital in FEDERATED_CLIENTS:
        print(f"\n=== {hospital} ===")
        subset = final_df[final_df["source_hospital"] == hospital]
        check = subset.groupby("split")[FINAL_CLASSES].sum()
        print(check)

        # Flag any class with 0 examples in either split
        zero_cols = check.columns[(check == 0).any(axis=0)]
        if len(zero_cols) > 0:
            print(f"  WARNING: these classes have 0 examples in train or test: {list(zero_cols)}")


if __name__ == "__main__":
    metadata_df = pd.read_csv(STAGE4_INPUT_PATH)
    print(f"Loaded {len(metadata_df)} records from Stage 4 output")

    final_df = split_all_clients(metadata_df)
    verify_class_balance(final_df)

    final_df.to_csv(STAGE6_OUTPUT_PATH, index=False)
    print(f"\nSaved: {STAGE6_OUTPUT_PATH}")
    print(f"Final shape: {final_df.shape}")
    print(final_df.groupby(["source_hospital", "split"]).size())
# ============ END: train_test_split.py ============
