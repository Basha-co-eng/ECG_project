import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 150
import pandas as pd
import numpy as np
import os

FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]
FIGDIR = r"C:\ECG_Project\training\figures"
COLORS = {"fedavg": "#2E86AB", "fedprox": "#E07A5F", "fedopt": "#6A994E"}
f1_std_combined = pd.read_csv(r"C:\ECG_Project\training\All excel files\f1_std_comparison.csv")
x = np.arange(len(FEDERATED_CLIENTS))
width = 0.25  # Reduced width so 3 bars fit comfortably side-by-side
fig, ax = plt.subplots(figsize=(10, 6))

fa_std = f1_std_combined[f1_std_combined["method"] == "FedAvg"].set_index("hospital")["f1_std"]
fp_std = f1_std_combined[f1_std_combined["method"] == "FedProx"].set_index("hospital")["f1_std"]
fo_std = f1_std_combined[f1_std_combined["method"] == "FedOpt"].set_index("hospital")["f1_std"]

# Offset positions sequentially: Left (-width), Center (x), Right (+width)
ax.bar(x - width, [fa_std[h] for h in FEDERATED_CLIENTS], width, label="FedAvg", color=COLORS["fedavg"])
ax.bar(x, [fp_std[h] for h in FEDERATED_CLIENTS], width, label="FedProx", color=COLORS["fedprox"])
ax.bar(x + width, [fo_std[h] for h in FEDERATED_CLIENTS], width, label="FedOpt", color=COLORS["fedopt"])

ax.set_xticks(x)
ax.set_xticklabels(FEDERATED_CLIENTS, rotation=20)
ax.set_ylabel("F1-STD (lower = more balanced across classes)")
ax.set_title("Per-Class F1 Standard Deviation — Long-Tail Robustness", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/f1_std_comparison.png", bbox_inches="tight")
plt.show()
