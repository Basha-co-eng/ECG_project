import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from ecgdataset_class_7 import ECGDataset
from resnet_class_6 import ResNet1D34

class ResNet1D34_MCDropout(ResNet1D34):
    """
    Same architecture as ResNet1D34, with one added Dropout layer before the
    final FC layer. Dropout has ZERO learnable parameters, so existing
    FedAvg/FedProx/Central checkpoints (trained WITHOUT this layer) load in
    directly via load_state_dict() -- no retraining needed.
    """
    def __init__(self, in_channels=12, num_classes=12, dropout_p=0.3):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.mc_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.mc_dropout(x)     # ONLY new line vs. the original forward()
        x = self.fc(x)
        return x


def enable_mc_dropout(model):
    """
    Puts the WHOLE model in eval mode (so BatchNorm uses running stats, not
    batch stats -- critical, since batch stats would themselves add noise we
    don't want) EXCEPT the dropout layer, which stays in train mode so it
    keeps randomly zeroing activations across repeated forward passes.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()




FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]
FINAL_CLASSES = ["AF", "IAVB", "LAD", "LBBB", "NSIVCB", "NSR", "PAC", "QAb", "RBBB", "SB", "STach", "TAb"]
SPLIT_CSV_PATH = r"C:\ECG_Project\training\ecg_metadata_train_test_split.csv"
PROCESSED_DIR = r"C:\ECG_Project\training\processed_signals"

MODEL_CHECKPOINTS = {
    "fedavg":  r"C:\ECG_Project\training\fedavg\global_model_round30.pt",
    "fedprox": r"C:\ECG_Project\training\fedprox_mu0.001\fedprox_global_model_round30.pt",
    "fedopt":  r"C:\ECG_Project\training\fedopt\fedopt_global_model_round30.pt",
}

N_MC_PASSES = 30          # standard MC Dropout sample count
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PATH = r"C:\ECG_Project\training\mc_dropout_uncertainty.csv"


def compute_mc_dropout_uncertainty(model, test_loader, device, n_passes=N_MC_PASSES):
    """
    For every record, runs n_passes stochastic forward passes (dropout active),
    and returns per-record: mean prediction (sigmoid probs) and uncertainty
    (std across passes) for each of the 12 classes.
    """
    enable_mc_dropout(model)

    all_record_ids = []
    all_means = []
    all_stds = []

    record_idx = 0
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            batch_size = signals.shape[0]

            pass_outputs = []
            for _ in range(n_passes):
                outputs = torch.sigmoid(model(signals))     # (batch, 12)
                pass_outputs.append(outputs.cpu().numpy())

            pass_outputs = np.stack(pass_outputs, axis=0)    # (n_passes, batch, 12)
            mean_pred = pass_outputs.mean(axis=0)             # (batch, 12)
            std_pred = pass_outputs.std(axis=0)                # (batch, 12) -- the uncertainty

            all_means.append(mean_pred)
            all_stds.append(std_pred)
            record_idx += batch_size

    all_means = np.concatenate(all_means, axis=0)
    all_stds = np.concatenate(all_stds, axis=0)
    return all_means, all_stds


def run_uncertainty_for_all(metadata_df):
    results = []

    for model_name, ckpt_path in MODEL_CHECKPOINTS.items():
        print(f"\n=== MC Dropout: {model_name} ===")
        model = ResNet1D34_MCDropout(in_channels=12, num_classes=12).to(device=DEVICE)
        state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state_dict, strict=False)   # strict=False: mc_dropout has no saved weights, that's expected
        model.to(DEVICE)

        for hospital in FEDERATED_CLIENTS:
            test_ds = ECGDataset(metadata_df, PROCESSED_DIR, hospital, "test")
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

            record_ids = test_ds.df["record_id"].tolist()
            means, stds = compute_mc_dropout_uncertainty(model, test_loader, DEVICE)

            print(f"  [{hospital}] n={len(record_ids)}  mean uncertainty (avg across classes)={stds.mean():.4f}")

            for i, rid in enumerate(record_ids):
                row = {"model": model_name, "hospital": hospital, "record_id": rid}
                for c_idx, cls in enumerate(FINAL_CLASSES):
                    row[f"{cls}_mean_pred"] = means[i, c_idx]
                    row[f"{cls}_uncertainty"] = stds[i, c_idx]
                results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}, shape={results_df.shape}")
    return results_df


metadata_df = pd.read_csv(SPLIT_CSV_PATH)
uncertainty_df = run_uncertainty_for_all(metadata_df)
