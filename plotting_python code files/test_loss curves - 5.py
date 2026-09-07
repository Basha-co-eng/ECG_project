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

COLORS = {"FedAvg": "#2E86AB", "FedProx": "#E07A5F", "FedOpt": "#BE398D", "Central": "#CFF800"}
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes = axes.flatten()
for hospital in FEDERATED_CLIENTS:
    fa = fedavg_hist[fedavg_hist["hospital"] == hospital].sort_values("round")
    axes[0].plot(fa["round"], fa["test_loss"], label=hospital, alpha=0.8)
    axes[0].set_title("FedAvg — Test Loss per Hospital", fontweight="bold")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Test Loss"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

for hospital in FEDERATED_CLIENTS:
    fp = fedprox_hist[fedprox_hist["hospital"] == hospital].sort_values("round")
    axes[1].plot(fp["round"], fp["test_loss"], label=hospital, alpha=0.8)
    axes[1].set_title("FedProx — Test Loss per Hospital", fontweight="bold")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Test Loss"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

for hospital in FEDERATED_CLIENTS:
    fo = fedopt_hist[fedopt_hist["hospital"] == hospital].sort_values("round")
    axes[2].plot(fo["round"], fo["test_loss"], label=hospital, alpha=0.8)
    axes[2].set_title("FedOpt — Test Loss per Hospital", fontweight="bold")
    axes[2].set_xlabel("Round"); axes[2].set_ylabel("Test Loss"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/loss_curves.png", bbox_inches="tight")
plt.show()
