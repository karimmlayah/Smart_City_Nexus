#!/usr/bin/env python3
"""
Entraîne le modèle CNN+LSTM combat / normal sur données synthétiques.

Usage:
  cd smart_crowd_safety_ai
  python scripts/train_fight.py --epochs 8

Métriques: accuracy, precision, recall (classe fight=1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.behavior.fight_cnn_lstm import FightCNNLSTM, save_checkpoint
from src.utils.config import FIGHT_FRAME_SIZE, FIGHT_SEQ_LEN, MODELS_DIR


class SyntheticFightDataset(Dataset):
    """
    Séquences artificielles:
    - label 0 (normal): frames corrélées (faible variation temporelle)
    - label 1 (fight): forte variation entre frames consécutives
    """

    def __init__(self, n_samples: int = 800, seq_len: int = FIGHT_SEQ_LEN, size: int = FIGHT_FRAME_SIZE):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.size = size
        self.labels = np.random.randint(0, 2, size=n_samples).astype(np.int64)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        label = self.labels[idx]
        if label == 0:
            base = np.random.randn(self.size, self.size, 3).astype(np.float32) * 0.15
            seq = np.stack([base + np.random.randn(self.size, self.size, 3).astype(np.float32) * 0.02 for _ in range(self.seq_len)], axis=0)
        else:
            seq = np.random.randn(self.seq_len, self.size, self.size, 3).astype(np.float32) * 0.35
        seq = np.clip(seq, -1, 1)
        # normaliser [0,1]
        seq = (seq - seq.min()) / (seq.max() - seq.min() + 1e-6)
        x = torch.from_numpy(seq).permute(0, 3, 1, 2).float()
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def train(epochs: int, batch_size: int, lr: float, out_path: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SyntheticFightDataset()
    n_val = int(len(ds) * 0.15)
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = FightCNNLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = 0.0
    metrics = {}

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)

        # validation + metrics
        model.eval()
        tp = fp = tn = fn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                for p, t in zip(pred.cpu().numpy(), y.cpu().numpy()):
                    if t == 1 and p == 1:
                        tp += 1
                    elif t == 0 and p == 1:
                        fp += 1
                    elif t == 0 and p == 0:
                        tn += 1
                    else:
                        fn += 1
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        print(f"Epoch {ep+1}/{epochs} loss={avg_loss:.4f} val_acc={acc:.3f} prec={prec:.3f} rec={rec:.3f}")

        if acc > best_val:
            best_val = acc
            metrics = {"accuracy": acc, "precision": prec, "recall": rec, "epoch": ep + 1}

    save_checkpoint(model, out_path, epoch=epochs, metrics=metrics)
    print(f"Modèle sauvegardé: {out_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "fight_cnn_lstm.pt"
    train(args.epochs, args.batch_size, args.lr, out_path)


if __name__ == "__main__":
    main()
