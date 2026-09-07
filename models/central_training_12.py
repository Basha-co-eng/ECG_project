from torch.utils.data import ConcatDataset, DataLoader
import pandas as pd
import os
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import torch
import torch.nn as nn
from resnet_class_6 import ResNet1D34
from ecgdataset_class_7 import ECGDataset, PROCESSED_DIR

FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]
SPLIT_CSV_PATH = r"C:\ECG_Project\training\ecg_metadata_train_test_split.csv"
CHECKPOINT_DIR = r"C:\ECG_Project\training"
NUM_EPOCHS = 30          # matches FedAvg/FedProx's 30-round budget for fair comparison
BATCH_SIZE = 16
CHECKPOINT_EVERY = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

metadata_df = pd.read_csv(SPLIT_CSV_PATH)

# Build individual per-hospital datasets (reuses your existing ECGDataset class),
# then pool them into ONE dataset — hospital identity is discarded here on purpose
train_datasets = [ECGDataset(metadata_df, PROCESSED_DIR, h, "train") for h in FEDERATED_CLIENTS]
test_datasets = [ECGDataset(metadata_df, PROCESSED_DIR, h, "test") for h in FEDERATED_CLIENTS]

pooled_train = ConcatDataset(train_datasets)
pooled_test = ConcatDataset(test_datasets)

pooled_train_loader = DataLoader(pooled_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
pooled_test_loader = DataLoader(pooled_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Pooled train size: {len(pooled_train)}  |  Pooled test size: {len(pooled_test)}")


def evaluate_central(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / len(test_loader)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, micro_f1, macro_f1


def find_latest_central_checkpoint(checkpoint_dir):
    import glob
    pattern = os.path.join(checkpoint_dir, "central_model_epoch*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None, 0
    epochs = [int(f.split("epoch")[-1].split(".pt")[0]) for f in checkpoints]
    latest_epoch = max(epochs)
    latest_path = os.path.join(checkpoint_dir, f"central_model_epoch{latest_epoch}.pt")
    try:
        torch.load(latest_path, map_location="cpu", weights_only=False)
        return latest_path, latest_epoch
    except Exception as e:
        print(f"Checkpoint failed to load ({e}) — starting fresh")
        return None, 0


def run_central_training():
    model = ResNet1D34(in_channels=12, num_classes=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    latest_ckpt, start_epoch = find_latest_central_checkpoint(CHECKPOINT_DIR)
    history = []

    if latest_ckpt:
        print(f"Resuming Central from checkpoint: {latest_ckpt} (epoch {start_epoch})")
        model.load_state_dict(torch.load(latest_ckpt, map_location=device, weights_only=False))
        history_path = os.path.join(CHECKPOINT_DIR, "central_history.csv")
        if os.path.exists(history_path):
            history = pd.read_csv(history_path).to_dict("records")
    else:
        print("No Central checkpoint found — starting fresh from epoch 1")

    if start_epoch >= NUM_EPOCHS:
        print(f"Central training already complete at epoch {start_epoch}.")
        return model, history

    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for signals, labels in pooled_train_loader:
            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)

            if not torch.isfinite(loss):
                print("  Skipping a batch with non-finite loss")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        test_loss, micro_f1, macro_f1 = evaluate_central(model, pooled_test_loader, device)

        print(f"Epoch {epoch}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  "
              f"micro_f1={micro_f1:.4f}  macro_f1={macro_f1:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss,
                         "micro_f1": micro_f1, "macro_f1": macro_f1})

        if epoch % CHECKPOINT_EVERY == 0 or epoch == NUM_EPOCHS:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"central_model_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            pd.DataFrame(history).to_csv(os.path.join(CHECKPOINT_DIR, "central_history.csv"), index=False)
            print(f"  Checkpoint + history saved at epoch {epoch}")

    return model, history


if __name__ == "__main__":
    model, history = run_central_training()
    print("\nCentral training run complete.")
