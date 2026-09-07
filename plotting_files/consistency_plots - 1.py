import matplotlib.pyplot as plt
import pandas as pd
phase_d_clean = pd.read_csv(r"C:\ECG_Project\training\All excel files\phase_d_consistency_uncertainty_CLEAN.csv")
# Plot 1 — scatter: uncertainty vs inconsistency, colored by model
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = {"fedavg": "#2E86AB", "fedprox": "#E07A5F", "fedopt": "#6A994E"}
FIGDIR = r"C:\ECG_Project\training\figures"
for ax, (col, label) in zip(axes, [("gradcam_consistency", "Grad-CAM"), ("shap_consistency", "SHAP")]):
    for model in phase_d_clean["model"].unique():
        sub = phase_d_clean[phase_d_clean["model"] == model]
        ax.scatter(sub["uncertainty_level"], 1 - sub[col], label=model, color=colors[model], s=70, alpha=0.8)
    ax.set_xlabel("MC Dropout Uncertainty (level)")
    ax.set_ylabel(f"{label} Inconsistency (1 - cosine similarity)")
    ax.set_title(f"Uncertainty vs. {label} Inconsistency")
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(rf"{FIGDIR}\phase_d_scatter.png", bbox_inches="tight")
plt.show()


# Plot 2 — degenerate class count per method
fig, ax = plt.subplots(figsize=(7, 5))
degenerate_counts = {"FedAvg": 0, "FedProx": 0, "FedOpt": 8}
ax.bar(degenerate_counts.keys(), degenerate_counts.values(), color=["#2E86AB", "#E07A5F", "#6A994E"])
ax.set_ylabel("Number of Classes with Degenerate (Zero-Signal) Grad-CAM")
ax.set_title("Explainability Collapse by Aggregation Strategy")
for i, v in enumerate(degenerate_counts.values()):
    ax.text(i, v + 0.15, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(rf"{FIGDIR}\degenerate_class_count.png", bbox_inches="tight")
plt.show()




#Grad CAM Plot of NSR, NSIVCB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
gradcam_data = np.load(r"C:\ECG_Project\training\stage10_gradcam\gradcam_heatmaps.npz")
FIGDIR = r"C:\ECG_Project\training\figures"
def plot_gradcam_overlay(model_name, hospital, class_name, lead_idx=1):
    """Overlays a Grad-CAM heatmap on one example waveform's Lead II."""
    key = f"{model_name}__{hospital}__{class_name}"
    if key not in gradcam_data:
        print(f"Not available: {key}")
        return
    heatmap = gradcam_data[key]   # (5000,) importance over time

    # grab one real example record from this hospital/class to plot against
    metadata_df = pd.read_csv(r"C:\ECG_Project\training\ecg_metadata_train_test_split.csv")
    subset = metadata_df[(metadata_df["source_hospital"] == hospital) &
                          (metadata_df[class_name] == 1) & (metadata_df["split"] == "test")]
    record_id = subset.iloc[0]["record_id"]
    signal = np.load(rf"C:\ECG_Project\training\processed_signals\{record_id}.npy")[lead_idx]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(signal, color="black", linewidth=0.8, label="ECG signal (Lead II)")
    ax.imshow(heatmap[np.newaxis, :], aspect="auto", cmap="Reds", alpha=0.4,
              extent=[0, len(signal), signal.min(), signal.max()])
    ax.set_title(f"Grad-CAM: {model_name} — {hospital} — {class_name} (record {record_id})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Timestep (500Hz, 10s window)")
    ax.set_ylabel("Normalized amplitude")
    plt.tight_layout()
    plt.savefig(rf"{FIGDIR}\gradcam_overlay_{model_name}_{hospital}_{class_name}.png", bbox_inches="tight")
    plt.show()

# A HIGH-consistency example
# the LOW-consistency FedProx/NSIVCB case (the actual finding)
plot_gradcam_overlay("fedavg", "chapman_shaoxing", "NSR")       # consistent, "normal" baseline
plot_gradcam_overlay("fedprox", "chapman_shaoxing", "NSIVCB")   # inconsistent case #1
plot_gradcam_overlay("fedprox", "ptb-xl", "NSIVCB")             # inconsistent case #2 -- same class, different hospital
