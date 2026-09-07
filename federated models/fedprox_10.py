import copy
from collections import OrderedDict
import pandas as pd
import os
import glob
import torch
import torch.nn as nn

import resnet_class_6 as resnet
import clients_utils_class_8 as client_utils


FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]
SPLIT_CSV_PATH = r"C:\ECG_Project\training\ecg_metadata_train_test_split.csv"
CHECKPOINT_DIR = r"C:\ECG_Project\training"

NUM_ROUNDS = 30          # SAME budget as FedAvg — required for a fair comparison
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
CHECKPOINT_EVERY = 5
MU = 0.001                # FedProx proximal term strength — standard starting value (Li et al., 2020)


def train_one_client_fedprox(model, global_weights, train_loader, device, epochs=1, lr=1e-3, mu=MU):
    """
    Same as train_one_client, PLUS a proximal term that penalizes the local
    model for drifting too far from the global weights it started this round with.
    This is the ONLY difference between FedAvg and FedProx.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Keep a frozen reference copy of the starting (global) weights for this round
    global_params = [p.clone().detach() for p in model.parameters()]

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)

            # --- Proximal term: mu/2 * ||local_weights - global_weights||^2 ---
            proximal_term = 0.0
            for local_p, global_p in zip(model.parameters(), global_params):
                proximal_term += (local_p - global_p).norm(2) ** 2
            loss = loss + (mu / 2) * proximal_term
            # --------------------------------------------------------------

            if not torch.isfinite(loss):
                print("  Skipping a batch with non-finite loss")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def federated_average(client_state_dicts, client_sizes):
    total_size = sum(client_sizes)
    avg_weights = OrderedDict()
    for key in client_state_dicts[0].keys():
        weighted_sum = sum(
            client_state_dicts[i][key].float() * (client_sizes[i] / total_size)
            for i in range(len(client_state_dicts))
        )
        avg_weights[key] = weighted_sum
    return avg_weights


def find_latest_checkpoint(checkpoint_dir, prefix="fedprox_global_model_round"):
    pattern = os.path.join(checkpoint_dir, f"{prefix}*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None, 0
    rounds = [int(f.split("round")[-1].split(".pt")[0]) for f in checkpoints]
    latest_round = max(rounds)
    latest_path = os.path.join(checkpoint_dir, f"{prefix}{latest_round}.pt")
    return latest_path, latest_round


def run_fedprox_training(client_loaders, device, num_rounds=NUM_ROUNDS, local_epochs=LOCAL_EPOCHS,
                          checkpoint_dir=CHECKPOINT_DIR, checkpoint_every=CHECKPOINT_EVERY, mu=MU):
    global_model = resnet.ResNet1D34(in_channels=12, num_classes=12).to(device)

    latest_ckpt, start_round = find_latest_checkpoint(checkpoint_dir)
    round_history = []

    if latest_ckpt:
        print(f"Found checkpoint: {latest_ckpt} (round {start_round}) — attempting to load")
        try:
            global_model.load_state_dict(torch.load(latest_ckpt, map_location=device, weights_only=False))
            print(f"Resumed successfully from round {start_round}")
            history_path = os.path.join(checkpoint_dir, "fedprox_round_history.csv")
            if os.path.exists(history_path):
                existing = pd.read_csv(history_path)
                for r in sorted(existing["round"].unique()):
                    round_history.append(
                        existing[existing["round"] == r].set_index("hospital")[
                            ["train_loss", "test_loss", "micro_f1", "precision", "recall",
                             "macro_f1", "macro_precision", "macro_recall"]
                        ].to_dict("index")
                    )
        except Exception as e:
            print(f"Checkpoint failed to load ({e}) — starting fresh from round 1 instead")
            start_round = 0
            round_history = []
    else:
        print("No FedProx checkpoint found — starting fresh from round 1")

    if start_round >= num_rounds:
        print(f"FedProx training already complete at round {start_round}.")
        return global_model, round_history

    for round_num in range(start_round + 1, num_rounds + 1):
        print(f"\n{'='*50}\nFEDPROX ROUND {round_num}/{num_rounds}\n{'='*50}")

        client_state_dicts = []
        client_sizes = []
        round_metrics = {}

        global_weights = copy.deepcopy(global_model.state_dict())

        for hospital in FEDERATED_CLIENTS:
            local_model = resnet.ResNet1D34(in_channels=12, num_classes=12).to(device)
            client_utils.set_model_parameters(local_model, global_weights)

            train_loss = client_utils.train_one_client_fedprox(
                local_model, global_weights, client_loaders[hospital]["train"], device,
                epochs=local_epochs, mu=mu
            )
            test_loss, f1, precision, recall, macro_f1, macro_precision, macro_recall = client_utils.evaluate_one_client(
                local_model, client_loaders[hospital]["test"], device
            )

            print(f"  [{hospital}] train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
                  f"micro_f1={f1:.4f} precision={precision:.4f} recall={recall:.4f} macro_f1: {macro_f1:.4f}, macro_precision: {macro_precision:.4f}, macro_recall: {macro_recall:.4f}")

            round_metrics[hospital] = {
                "train_loss": train_loss, "test_loss": test_loss,
                "micro_f1": f1, "precision": precision, "recall": recall,"macro_f1": macro_f1, "macro_precision": macro_precision, "macro_recall": macro_recall
            }
            client_state_dicts.append(client_utils.get_model_parameters(local_model))
            client_sizes.append(client_loaders[hospital]["train_size"])

        new_global_weights = client_utils.federated_average(client_state_dicts, client_sizes)
        client_utils.set_model_parameters(global_model, new_global_weights)
        round_history.append(round_metrics)
        print(f"\nFedProx Round {round_num} complete. Global model updated.")

        if round_num % checkpoint_every == 0 or round_num == num_rounds:
            ckpt_path = os.path.join(checkpoint_dir, f"fedprox_global_model_round{round_num}.pt")
            torch.save(global_model.state_dict(), ckpt_path)

            history_rows = []
            for r, metrics in enumerate(round_history, start=1):
                for hospital, vals in metrics.items():
                    history_rows.append({"round": r, "hospital": hospital, **vals})
            pd.DataFrame(history_rows).to_csv(
                os.path.join(checkpoint_dir, "fedprox_round_history.csv"), index=False
            )
            print(f"  FedProx checkpoint + history saved at round {round_num}")

    return global_model, round_history


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metadata_df = pd.read_csv(SPLIT_CSV_PATH)
    client_loaders = client_utils.build_client_loaders(metadata_df, FEDERATED_CLIENTS, batch_size=BATCH_SIZE)

    global_model, round_history = run_fedprox_training(client_loaders, device)

    print("\n\nFedProx training run complete (30 rounds).")
# ============ END: fedprox.py ============
