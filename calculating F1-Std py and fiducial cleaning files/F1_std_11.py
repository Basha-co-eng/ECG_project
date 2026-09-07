import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import resnet_class_6 as resnet
import clients_utils_class_8 as client_utils


FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]
FINAL_CLASSES = ["AF", "IAVB", "LAD", "LBBB", "NSIVCB", "NSR", "PAC", "QAb", "RBBB", "SB", "STach", "TAb"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_per_class_f1(model, test_loader, device):
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

    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    return dict(zip(FINAL_CLASSES, per_class_f1))


def compute_f1_std_table(checkpoint_path, method_name):
    model = resnet.ResNet1D34(in_channels=12, num_classes=12).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))

    rows = []
    for hospital in FEDERATED_CLIENTS:
        per_class = get_per_class_f1(model, client_utils.client_loaders[hospital]["test"], device)
        f1_values = list(per_class.values())

        row = {"method": method_name, "hospital": hospital}
        row.update(per_class)
        row["macro_f1_check"] = np.mean(f1_values)   # sanity cross-check vs. your existing macro_f1
        row["f1_std"] = np.std(f1_values)             # FedCVD's exact metric
        rows.append(row)

    return pd.DataFrame(rows)


fedavg_std_df = compute_f1_std_table(r"C:\ECG_Project\training\fedavg\global_model_round30.pt", "FedAvg")
fedprox_std_df = compute_f1_std_table(r"C:\ECG_Project\training\fedprox_mu0.001\fedprox_global_model_round30.pt", "FedProx")
fedopt_std_df = compute_f1_std_table(r"C:\ECG_Project\training\fedopt\fedopt_global_model_round30.pt", "FedOpt")

f1_std_combined = pd.concat([fedavg_std_df, fedprox_std_df, fedopt_std_df], ignore_index=True)
f1_std_combined.to_csv(r"C:\ECG_Project\training\f1_std_comparison.csv", index=False)

print(f1_std_combined[["method", "hospital", "f1_std", "macro_f1_check"]])
