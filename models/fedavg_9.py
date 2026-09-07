# ============ START: fedavg.py (30-round run, resumable) ============
import torch
import copy
from collections import OrderedDict
import pandas as pd
import os
import glob
import numpy as np
import resnet_class_6 as resnet
import ecgdataset_class_7 as ecg_dataset
import clients_utils_class_8 as client_utils
from sklearn.metrics import f1_score, precision_score, recall_score


FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]
SPLIT_CSV_PATH = r"C:\ECG_Project\training\ecg_metadata_train_test_split.csv"
CHECKPOINT_DIR = r"C:\ECG_Project\training"

NUM_ROUNDS = 30
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
CHECKPOINT_EVERY = 5  # save + allow resume every 5 rounds
CHECKPOINT_ROUNDS = [5, 10, 15, 20, 25, 30]

def federated_average(client_state_dicts, client_sizes):
    """
    Weighted average of multiple clients' model weights.
    A hospital with more training data gets proportionally more influence
    (standard FedAvg, McMahan et al.).
    """
    total_size = sum(client_sizes)
    avg_weights = OrderedDict()
    for key in client_state_dicts[0].keys():
        weighted_sum = sum(
            client_state_dicts[i][key].float() * (client_sizes[i] / total_size)
            for i in range(len(client_state_dicts))
        )
        avg_weights[key] = weighted_sum
    return avg_weights


def find_latest_checkpoint(checkpoint_dir):
    """Finds the highest-round checkpoint saved so far, if any."""
    pattern = os.path.join(checkpoint_dir, "global_model_round*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None, 0
    rounds = [int(f.split("round")[-1].split(".pt")[0]) for f in checkpoints]
    latest_round = max(rounds)
    latest_path = os.path.join(checkpoint_dir, f"global_model_round{latest_round}.pt")
    return latest_path, latest_round


def run_federated_training(client_loaders, device, num_rounds=NUM_ROUNDS, local_epochs=LOCAL_EPOCHS,
                            checkpoint_dir=CHECKPOINT_DIR, checkpoint_every=CHECKPOINT_EVERY):
    global_model = resnet.ResNet1D34(in_channels=12, num_classes=12).to(device)

    # RESUME LOGIC: pick up from the last saved round instead of starting over
    latest_ckpt, start_round = find_latest_checkpoint(checkpoint_dir)
    round_history = []

    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt} (round {start_round})")
        global_model.load_state_dict(torch.load(latest_ckpt, map_location=device))

        history_path = os.path.join(checkpoint_dir, "fedavg_round_history.csv")
        if os.path.exists(history_path):
            existing = pd.read_csv(history_path)
            for r in sorted(existing["round"].unique()):
                round_history.append(
                    existing[existing["round"] == r].set_index("hospital")[
                        ["train_loss", "test_loss", "micro_f1", "precision", "recall"]
                    ].to_dict("index")
                )
    else:
        print("No checkpoint found — starting fresh from round 1")

    if start_round >= num_rounds:
        print(f"Already completed {start_round} rounds (target was {num_rounds}). Nothing to do.")
        return global_model, round_history

    for round_num in range(start_round + 1, num_rounds + 1):
        print(f"\n{'='*50}\nROUND {round_num}/{num_rounds}\n{'='*50}")

        client_state_dicts = []
        client_sizes = []
        round_metrics = {}

        global_weights = copy.deepcopy(global_model.state_dict())

        for hospital in FEDERATED_CLIENTS:
            local_model = resnet.ResNet1D34(in_channels=12, num_classes=12).to(device)
            client_utils.set_model_parameters(local_model, global_weights)

            train_loss = client_utils.train_one_client(
                local_model, client_loaders[hospital]["train"], device, epochs=local_epochs
            )
            test_loss, f1, precision, recall = client_utils.evaluate_one_client(
                local_model, client_loaders[hospital]["test"], device
            )

            print(f"  [{hospital}] train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
                  f"micro_f1={f1:.4f} precision={precision:.4f} recall={recall:.4f}")

            round_metrics[hospital] = {
                "train_loss": train_loss, "test_loss": test_loss,
                "micro_f1": f1, "precision": precision, "recall": recall,
            }
            client_state_dicts.append(client_utils.get_model_parameters(local_model))
            client_sizes.append(client_loaders[hospital]["train_size"])

        new_global_weights = client_utils.federated_average(client_state_dicts, client_sizes)
        client_utils.set_model_parameters(global_model, new_global_weights)
        round_history.append(round_metrics)
        print(f"\nRound {round_num} complete. Global model updated.")

        # Save checkpoint + history every N rounds, and always on the final round
        if round_num % checkpoint_every == 0 or round_num == num_rounds:
            ckpt_path = os.path.join(checkpoint_dir, f"global_model_round{round_num}.pt")
            torch.save(global_model.state_dict(), ckpt_path)

            history_rows = []
            for r, metrics in enumerate(round_history, start=1):
                for hospital, vals in metrics.items():
                    history_rows.append({"round": r, "hospital": hospital, **vals})
            pd.DataFrame(history_rows).to_csv(
                os.path.join(checkpoint_dir, "fedavg_round_history.csv"), index=False
            )
            print(f"  Checkpoint + history saved at round {round_num}")

    return global_model, round_history


def evaluate_checkpoint(model, test_loader, device):
    """Runs one hospital's test set through the model. Pure evaluation — no training."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            outputs = model(signals)
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return {
        "micro_f1": f1_score(all_labels, all_preds, average="micro", zero_division=0),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "micro_precision": precision_score(all_labels, all_preds, average="micro", zero_division=0),
        "macro_precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "micro_recall": recall_score(all_labels, all_preds, average="micro", zero_division=0),
        "macro_recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
    }




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metadata_df = pd.read_csv(SPLIT_CSV_PATH)
    client_loaders = client_utils.build_client_loaders(metadata_df, FEDERATED_CLIENTS, batch_size=BATCH_SIZE)

    global_model, round_history = run_federated_training(client_loaders, device)

    print("\n\nFederated training run complete.")
    print(f"Final checkpoint and history are already saved in {CHECKPOINT_DIR}")


    all_results = []

    for round_num in CHECKPOINT_ROUNDS:
        ckpt_path = f"{CHECKPOINT_DIR}\\global_model_round{round_num}.pt"
        print(f"\n--- Evaluating checkpoint: round {round_num} ---")

        model = resnet.ResNet1D34(in_channels=12, num_classes=12).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        for hospital in FEDERATED_CLIENTS:
            metrics = client_utils.evaluate_checkpoint(model, client_loaders[hospital]["test"], device)
            print(f"  {hospital}: micro_f1={metrics['micro_f1']:.4f}  macro_f1={metrics['macro_f1']:.4f}")

            all_results.append({
                "round": round_num,
                "hospital": hospital,
                **metrics
            })

    results_df = pd.DataFrame(all_results)
    out_path = f"{CHECKPOINT_DIR}\\fedavg_micro_macro_full_history.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(results_df)
# ============ END: fedavg.py ============
