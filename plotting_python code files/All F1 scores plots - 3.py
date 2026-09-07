import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 150
import pandas as pd
import numpy as np
import os

FIGDIR = r"C:\ECG_Project\training\figures"
os.makedirs(FIGDIR, exist_ok=True)

FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]

fedavg_hist = pd.read_csv(r"C:\ECG_Project\training\fedavg\fedavg_round_history.csv")
fedprox_hist = pd.read_csv(r"C:\ECG_Project\training\fedprox_mu0.001\fedprox_round_history.csv")
fedopt_hist = pd.read_csv(r"C:\ECG_Project\training\fedopt\fedopt_round_history.csv")
central_hist = pd.read_csv(r"C:\ECG_Project\training\central_model\central_history.csv")

COLORS = {"FedAvg": "#2E86AB", "FedProx": "#E07A5F", "FedOpt": "#BE398D", "Central": "#CFF800"}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

# Panels 0-4: the 5 hospitals, federated methods only (Central has no per-hospital breakdown)
for i, hospital in enumerate(FEDERATED_CLIENTS):
    ax = axes[i]
    fa = fedavg_hist[fedavg_hist["hospital"] == hospital].sort_values("round")
    fp = fedprox_hist[fedprox_hist["hospital"] == hospital].sort_values("round")
    fo = fedopt_hist[fedopt_hist["hospital"] == hospital].sort_values("round")

    ax.plot(fa["round"], fa["micro_f1"], label="FedAvg", color=COLORS["FedAvg"], linewidth=2)
    ax.plot(fp["round"], fp["micro_f1"], label="FedProx (\u03bc=0.001)", color=COLORS["FedProx"], linewidth=2)
    ax.plot(fo["round"], fo["micro_f1"], label="FedOpt", color=COLORS["FedOpt"], linewidth=2)

    ax.set_title(hospital, fontsize=12, fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Micro-F1")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

# Panel 5 (the one you were turning off): Central's own epoch-based curve, since it's pooled, not per-hospital
ax6 = axes[5]
central_sorted = central_hist.sort_values("epoch")
ax6.plot(central_sorted["epoch"], central_sorted["micro_f1"], label="Central", color=COLORS["Central"], linewidth=2)
ax6.axvline(x=15, color="black", linestyle="--", linewidth=1, alpha=0.6, label="Reported checkpoint (epoch 15)")
ax6.set_title("Centralized (Pooled)", fontsize=12, fontweight="bold")
ax6.set_xlabel("Epoch")
ax6.set_ylabel("Micro-F1")
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3)

plt.suptitle("FedAvg vs FedProx vs FedOpt vs Centralized \u2014 Micro-F1 Convergence", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/micro_f1_convergence_all4.png", bbox_inches="tight")
plt.show()
