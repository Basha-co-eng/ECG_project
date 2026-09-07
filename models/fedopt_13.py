# ============ START: fedopt.py ============
import torch
import torch.nn as nn
import copy
from collections import OrderedDict
import pandas as pd
import os
import glob
import resnet_class_6 as resnet
import clients_utils_class_8 as client_utils

FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]
SPLIT_CSV_PATH = r"C:\ECG_Project\training\ecg_metadata_train_test_split.csv"
CHECKPOINT_DIR = r"C:\ECG_Project\training\fedopt"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

NUM_ROUNDS = 30
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
CHECKPOINT_EVERY = 5

# FedAdam server-side hyperparameters (standard defaults, Reddi et al. 2020)
SERVER_LR = 0.01
BETA1 = 0.9
BETA2 = 0.99
EPSILON = 1e-3


def fedavg_aggregate(client_state_dicts, client_sizes):
    """Standard FedAvg weighted average -- same as before, used as the RAW aggregation
    before FedAdam's momentum adjustment is applied on top."""
    total_size = sum(client_sizes)
    avg_weights = OrderedDict()
    for key in client_state_dicts[0].keys():
        weighted_sum = sum(
            client_state_dicts[i][key].float() * (client_sizes[i] / total_size)
            for i in range(len(client_state_dicts))
        )
        avg_weights[key] = weighted_sum
    return avg_weights


def fedadam_update(global_weights, avg_weights, m, v, server_lr=SERVER_LR, beta1=BETA1, beta2=BETA2, eps=EPSILON):
    """
    FedAdam server-side update. Treats (avg_weights - global_weights) as a
    'pseudo-gradient' from the server's perspective, then applies Adam-style
    momentum + adaptive scaling on top of it -- same idea as Adam for local
    training, just applied to the AGGREGATION step instead.
    """
    new_global = OrderedDict()
    new_m = OrderedDict()
    new_v = OrderedDict()

    for key in global_weights.keys():
        delta = avg_weights[key].float() - global_weights[key].float()

        new_m[key] = beta1 * m[key] + (1 - beta1) * delta
        new_v[key] = beta2 * v[key] + (1 - beta2) * (delta ** 2)

        new_global[key] = global_weights[key].float() + server_lr * new_m[key] / (torch.sqrt(new_v[key]) + eps)

    return new_global, new_m, new_v


def find_latest_checkpoint(checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, "fedopt_global_model_round*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None, 0
    rounds = [int(f.split("round")[-1].split(".pt")[0]) for f in checkpoints]
    latest_round = max(rounds)
    latest_path = os.path.join(checkpoint_dir, f"fedopt_global_model_round{latest_round}.pt")
    try:
        torch.load(latest_path, map_location="cpu", weights_only=False)
        return latest_path, latest_round
    except Exception as e:
        print(f"Checkpoint failed to load ({e}) — starting fresh")
        return None, 0


def run_fedopt_training(client_loaders, device, num_rounds=NUM_ROUNDS, local_epochs=LOCAL_EPOCHS,
                         checkpoint_dir=CHECKPOINT_DIR, checkpoint_every=CHECKPOINT_EVERY):
    global_model = resnet.ResNet1D34(in_channels=12, num_classes=12).to(device)

    latest_ckpt, start_round = find_latest_checkpoint(checkpoint_dir)
    round_history = []

    # Initialize server-side momentum/variance accumulators at zero
    m = OrderedDict((k, torch.zeros_like(v).float()) for k, v in global_model.state_dict().items())
    v_acc = OrderedDict((k, torch.zeros_like(val).float()) for k, val in global_model.state_dict().items())

    if latest_ckpt:
        print(f"Resuming FedOpt from checkpoint: {latest_ckpt} (round {start_round})")
        global_model.load_state_dict(torch.load(latest_ckpt, map_location=device, weights_only=False))
        history_path = os.path.join(checkpoint_dir, "fedopt_round_history.csv")
        if os.path.exists(history_path):
            existing = pd.read_csv(history_path)
            for r in sorted(existing["round"].unique()):
                round_history.append(
                    existing[existing["round"] == r].set_index("hospital")[
                        ["train_loss", "test_loss", "micro_f1", "precision", "recall",
                         "macro_f1", "macro_precision", "macro_recall"]
                    ].to_dict("index")
                )
        # NOTE: m/v accumulators reset to zero on resume -- a minor approximation,
        # acceptable since momentum re-warms quickly over a few rounds
    else:
        print("No FedOpt checkpoint found — starting fresh from round 1")

    if start_round >= num_rounds:
        print(f"FedOpt training already complete at round {start_round}.")
        return global_model, round_history

    for round_num in range(start_round + 1, num_rounds + 1):
        print(f"\n{'='*50}\nFEDOPT ROUND {round_num}/{num_rounds}\n{'='*50}")

        client_state_dicts = []
        client_sizes = []
        round_metrics = {}

        global_weights = copy.deepcopy(global_model.state_dict())

        for hospital in FEDERATED_CLIENTS:
            local_model = resnet.ResNet1D34(in_channels=12, num_classes=12).to(device)
            client_utils.set_model_parameters(local_model, global_weights)

            # Local training is PLAIN FedAvg-style (no proximal term) -- FedOpt's
            # novelty is entirely in the server aggregation step, not local training
            train_loss = client_utils.train_one_client(
                local_model, client_loaders[hospital]["train"], device, epochs=local_epochs
            )
            test_loss, f1, precision, recall, macro_f1, macro_precision, macro_recall = client_utils.evaluate_one_client(
                local_model, client_loaders[hospital]["test"], device
            )

            print(f"  [{hospital}] train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
                  f"micro_f1={f1:.4f} macro_f1={macro_f1:.4f}")

            round_metrics[hospital] = {
                "train_loss": train_loss, "test_loss": test_loss,
                "micro_f1": f1, "precision": precision, "recall": recall,
                "macro_f1": macro_f1, "macro_precision": macro_precision, "macro_recall": macro_recall
            }
            client_state_dicts.append(client_utils.get_model_parameters(local_model))
            client_sizes.append(client_loaders[hospital]["train_size"])

        # Step 1: standard FedAvg aggregation
        avg_weights = fedavg_aggregate(client_state_dicts, client_sizes)
        # Step 2: FedAdam momentum adjustment ON TOP of the averaged weights
        new_global_weights, m, v_acc = fedadam_update(global_weights, avg_weights, m, v_acc)

        client_utils.set_model_parameters(global_model, new_global_weights)
        round_history.append(round_metrics)
        print(f"\nFedOpt Round {round_num} complete. Global model updated.")

        if round_num % checkpoint_every == 0 or round_num == num_rounds:
            ckpt_path = os.path.join(checkpoint_dir, f"fedopt_global_model_round{round_num}.pt")
            torch.save(global_model.state_dict(), ckpt_path)

            history_rows = []
            for r, metrics in enumerate(round_history, start=1):
                for hospital, vals in metrics.items():
                    history_rows.append({"round": r, "hospital": hospital, **vals})
            pd.DataFrame(history_rows).to_csv(
                os.path.join(checkpoint_dir, "fedopt_round_history.csv"), index=False
            )
            print(f"  FedOpt checkpoint + history saved at round {round_num}")

    return global_model, round_history


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metadata_df = pd.read_csv(SPLIT_CSV_PATH)
    client_loaders = client_utils.build_client_loaders(metadata_df, FEDERATED_CLIENTS, batch_size=BATCH_SIZE)

    global_model, round_history = run_fedopt_training(client_loaders, device)
    print("\n\nFedOpt training run complete.")
# ============ END: fedopt.py ============
