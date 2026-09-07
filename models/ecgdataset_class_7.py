# ============ START: ecg_dataset.py ============
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import resnet_class_6 as resnet  # Import the ResNet1D34 model from resnet_class_6.py

FINAL_CLASSES = ["AF", "IAVB", "LAD", "LBBB", "NSIVCB", "NSR", "PAC", "QAb", "RBBB", "SB", "STach", "TAb"]

PROCESSED_DIR = r"C:\ECG_Project\training\processed_signals"
SPLIT_CSV_PATH = r"C:\ECG_Project\training\ecg_metadata_train_test_split.csv"


class ECGDataset(Dataset):
    """
    Loads preprocessed (12, 5000) .npy signals + multi-label targets
    for one specific hospital client and split (train/test).

    Defensive by design: filters out any record whose .npy file doesn't
    actually exist on disk, at __init__ time — not mid-training. This is
    the fix for the FileNotFoundError bug from Stage 4; baking it in here
    means it can never resurface, even if a metadata CSV gets stale again.
    """
    def __init__(self, metadata_df: pd.DataFrame, processed_dir: str,
                 source_hospital: str, split: str):
        candidate_df = metadata_df[
            (metadata_df["source_hospital"] == source_hospital) &
            (metadata_df["split"] == split)
        ].reset_index(drop=True)

        exists_mask = candidate_df["record_id"].apply(
            lambda rid: Path(f"{processed_dir}/{rid}.npy").exists()
        )
        dropped = (~exists_mask).sum()
        if dropped > 0:
            print(f"  [{source_hospital}/{split}] Skipping {dropped} records with missing .npy files")

        self.df = candidate_df[exists_mask].reset_index(drop=True)
        self.processed_dir = processed_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = row["record_id"]

        signal = np.load(f"{self.processed_dir}/{record_id}.npy")
        label = row[FINAL_CLASSES].values.astype(np.float32)

        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def build_client_loaders(metadata_df: pd.DataFrame, federated_clients: list,
                          processed_dir: str = PROCESSED_DIR, batch_size: int = 16):
    """
    Builds one train + test DataLoader pair per hospital client.
    Returns a dict: { hospital_name: {"train": loader, "test": loader, "train_size": int} }
    """
    client_loaders = {}
    for hospital in federated_clients:
        train_ds = ECGDataset(metadata_df, processed_dir, hospital, "train")
        test_ds = ECGDataset(metadata_df, processed_dir, hospital, "test")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        client_loaders[hospital] = {
            "train": train_loader,
            "test": test_loader,
            "train_size": len(train_ds),
        }
        print(f"{hospital}: {len(train_ds)} train / {len(test_ds)} test")

    return client_loaders


if __name__ == "__main__":

    FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]

    metadata_df = pd.read_csv(SPLIT_CSV_PATH)
    client_loaders = build_client_loaders(metadata_df, FEDERATED_CLIENTS)

    # End-to-end sanity check: pull one real batch through the model
    model = resnet.ResNet1D34(in_channels=12, num_classes=12)

    test_hospital = "georgia"
    batch_signals, batch_labels = next(iter(client_loaders[test_hospital]["train"]))
    print(f"\nBatch signals shape: {batch_signals.shape}")   # expect (16, 12, 5000)
    print(f"Batch labels shape: {batch_labels.shape}")       # expect (16, 12)

    test_output = model(batch_signals)
    print(f"Model output shape on real batch: {test_output.shape}")   # expect (16, 12)
# ============ END: ecg_dataset.py ============
