# ============ START: gradcam.py ============
import os
from anyio import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from resnet_class_6 import ResNet1D34
from clients_utils_class_8 import ECGDataset



# ---------------- Config ----------------
FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]
FINAL_CLASSES = ["AF", "IAVB", "LAD", "LBBB", "NSIVCB", "NSR", "PAC", "QAb", "RBBB", "SB", "STach", "TAb"]
CHECKPOINT_DIR = r"C:\ECG_Project\training"
SPLIT_CSV_PATH = r"C:\ECG_Project\training\ecg_metadata_train_test_split.csv"
PROCESSED_DIR = r"C:\ECG_Project\training\processed_signals"

# >>> UPDATE THESE TWO PATHS to match your new folder locations exactly <
MODEL_CHECKPOINTS = {
    "fedavg":  os.path.join(CHECKPOINT_DIR, "fedavg", "global_model_round30.pt"),
    "fedprox": os.path.join(CHECKPOINT_DIR, "fedprox_mu0.001", "fedprox_global_model_round30.pt"),
    "fedopt":  os.path.join(CHECKPOINT_DIR, "fedopt", "fedopt_global_model_round30.pt"),
}

OUTPUT_DIR = os.path.join(CHECKPOINT_DIR, "stage10_gradcam")
os.makedirs(OUTPUT_DIR, exist_ok=True)
HEATMAP_NPZ_PATH = os.path.join(OUTPUT_DIR, "gradcam_heatmaps.npz")
SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, "gradcam_summary.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_RECORDS_PER_CELL = 5


class GradCAM1D:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x: torch.Tensor, target_class_idx: int, signal_length: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        output = self.model(x)
        score = output[0, target_class_idx]
        score.backward()

        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam = cam.unsqueeze(1)
        cam = F.interpolate(cam, size=signal_length, mode="linear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam


def load_model(checkpoint_path: str) -> torch.nn.Module:
    model = ResNet1D34(in_channels=12, num_classes=len(FINAL_CLASSES)).to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    # NOTE: parameters are NOT frozen here — Grad-CAM's backward() requires an
    # intact autograd graph. eval() mode alone (disabling dropout/batchnorm
    # updates) is sufficient and correct; freezing requires_grad breaks backward().
    return model


def run_gradcam_for_model(model_name: str, checkpoint_path: str, metadata_df: pd.DataFrame):
    print(f"\n=== Grad-CAM: {model_name} ({checkpoint_path}) ===")
    model = load_model(checkpoint_path)
    cam_engine = GradCAM1D(model, target_layer=model.layer4)

    results = {}
    counts = {}
    summary_rows = []

    for hospital in FEDERATED_CLIENTS:
        test_ds = ECGDataset(metadata_df, PROCESSED_DIR, hospital, "test")
        if len(test_ds) == 0:
            print(f"  [{hospital}] no test records, skipping")
            continue

        loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

        for class_idx, class_name in enumerate(FINAL_CLASSES):
            key = f"{model_name}__{hospital}__{class_name}"
            heatmap_sum = np.zeros(5000, dtype=np.float64)
            n_used = 0

            for signal, label in loader:
                if label[0, class_idx].item() != 1.0:
                    continue

                signal = signal.to(DEVICE)
                cam = cam_engine.generate(signal, target_class_idx=class_idx, signal_length=signal.shape[-1])
                heatmap_sum += cam
                n_used += 1

            if n_used >= MIN_RECORDS_PER_CELL:
                results[key] = (heatmap_sum / n_used).astype(np.float32)
            counts[key] = n_used
            summary_rows.append({
                "model": model_name, "hospital": hospital, "class": class_name,
                "n_records_used": n_used,
                "included": n_used >= MIN_RECORDS_PER_CELL
            })
            if n_used > 0:
                print(f"  [{hospital}/{class_name}] n={n_used}"
                      + ("" if n_used >= MIN_RECORDS_PER_CELL else "  (below min, excluded)"))

    return results, summary_rows


if __name__ == "__main__":
    metadata_df = pd.read_csv(SPLIT_CSV_PATH)

    all_heatmaps = {}
    all_summary_rows = []

    for model_name, ckpt_path in MODEL_CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"WARNING: checkpoint not found for {model_name}: {ckpt_path} — skipping")
            continue
        heatmaps, summary_rows = run_gradcam_for_model(model_name, ckpt_path, metadata_df)
        all_heatmaps.update(heatmaps)
        all_summary_rows.extend(summary_rows)

    np.savez_compressed(HEATMAP_NPZ_PATH, **all_heatmaps)
    pd.DataFrame(all_summary_rows).to_csv(SUMMARY_CSV_PATH, index=False)

    print(f"\nSaved {len(all_heatmaps)} heatmaps to: {HEATMAP_NPZ_PATH}")
    print(f"Saved summary to: {SUMMARY_CSV_PATH}")
# ============ END: gradcam.py ============
