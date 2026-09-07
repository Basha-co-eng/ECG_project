# ============ START: client_utils.py ============
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import torch

def get_model_parameters(model):
    """Extract model weights as a plain state_dict (for aggregation)."""
    return model.state_dict()


def set_model_parameters(model, state_dict):
    """Load a state_dict back into the model."""
    model.load_state_dict(state_dict, strict=True)


def train_one_client(model, train_loader, device, epochs=1, lr=1e-3):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)

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


def evaluate_one_client(model, test_loader, device):
    model.to(device)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    micro_precision = precision_score(all_labels, all_preds, average="micro", zero_division=0)
    micro_recall = recall_score(all_labels, all_preds, average="micro", zero_division=0)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)      # NEW
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)  # NEW
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)        # NEW

    return avg_loss, micro_f1, micro_precision, micro_recall, macro_f1, macro_precision, macro_recall
