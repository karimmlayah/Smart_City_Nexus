"""
Classification combat / normal : CNN + LSTM sur séquence de frames.

Entrée: T frames RGB redimensionnées (FIGHT_SEQ_LEN x H x W).
Sortie: logits [normal, fight] -> softmax pour P(fight).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import FIGHT_FRAME_SIZE, FIGHT_SEQ_LEN


def preprocess_frame_for_fight(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR -> RGB, resize carré, float32 [0,1]."""
    import cv2

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (FIGHT_FRAME_SIZE, FIGHT_FRAME_SIZE))
    return rgb.astype(np.float32) / 255.0


class FrameCNN(nn.Module):
    """Petit CNN sur une frame."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        z = self.conv(x)
        z = z.flatten(1)
        return self.fc(z)


class FightCNNLSTM(nn.Module):
    def __init__(
        self,
        seq_len: int = FIGHT_SEQ_LEN,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.cnn = FrameCNN(embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=1)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (B, T, C, H, W)
        """
        b, t, c, h, w = seq.shape
        x = seq.view(b * t, c, h, w)
        emb = self.cnn(x)
        emb = emb.view(b, t, -1)
        out, _ = self.lstm(emb)
        last = out[:, -1, :]
        return self.head(last)

    @torch.no_grad()
    def predict_fight_probability(self, frame_sequence: np.ndarray) -> float:
        """
        frame_sequence: (T, H, W, 3) float32 RGB [0,1]
        Retourne P(classe fight) approximée (index 1).
        """
        self.eval()
        if frame_sequence.shape[0] < self.seq_len:
            # pad par répétition de la dernière frame
            pad = self.seq_len - frame_sequence.shape[0]
            last = frame_sequence[-1:]
            frame_sequence = np.concatenate([frame_sequence, np.repeat(last, pad, axis=0)], axis=0)
        elif frame_sequence.shape[0] > self.seq_len:
            frame_sequence = frame_sequence[-self.seq_len :]

        x = torch.from_numpy(frame_sequence).permute(0, 3, 1, 2).unsqueeze(0)
        logits = self.forward(x)
        prob = F.softmax(logits, dim=-1)[0, 1].item()
        return float(prob)

    @staticmethod
    def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> "FightCNNLSTM":
        try:
            ckpt = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=map_location)
        model = FightCNNLSTM()
        model.load_state_dict(ckpt["model_state_dict"])
        return model


def save_checkpoint(model: FightCNNLSTM, path: Path, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )
