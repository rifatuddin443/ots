"""Optional torch-LSTM classifier drop-in.

Used automatically by the experiment harness when torch is installed; otherwise
the numpy :class:`~fhew.baselines.learned.LearnedClassifier` is used. Kept as a
separate module so the core package has no deep-learning dependency, matching
the paper's reuse plan (the LSTM checkpoint becomes a journal baseline).
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised only when torch is present
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def torch_available() -> bool:
    return _HAS_TORCH


if _HAS_TORCH:  # pragma: no cover - only when torch installed

    class _LSTMNet(nn.Module):
        def __init__(self, n_channels: int, n_classes: int, hidden: int = 64):
            super().__init__()
            self.embed = nn.Embedding(n_channels + 1, 16)  # +1 for "empty slot"
            self.lstm = nn.LSTM(16, hidden, batch_first=True)
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, x, lengths=None):
            e = self.embed(x)
            out, _ = self.lstm(e)
            return self.head(out[:, -1, :])

    class LSTMClassifier:
        label = "lstm"

        def __init__(self, n_channels: int, classes: list[str], hidden: int = 64,
                     lr: float = 1e-2, device: str = "cpu"):
            self.n_channels = n_channels
            self.classes = list(classes)
            self.device = device
            self.net = _LSTMNet(n_channels, len(classes), hidden).to(device)
            self.lr = lr

        def _encode(self, obs_list):
            # dominant detected channel per slot, or n_channels for an empty slot
            seq = []
            for o in obs_list:
                if len(o.detections) == 0:
                    seq.append(self.n_channels)
                else:
                    seq.append(o.detections[0].channel)
            return torch.tensor(seq, dtype=torch.long)

        def fit(self, episodes, labels, epochs: int = 15, batch: int = 64):
            opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            lossf = nn.CrossEntropyLoss()
            y = torch.tensor(labels, dtype=torch.long)
            seqs = [self._encode(e) for e in episodes]
            L = max(len(s) for s in seqs)
            X = torch.full((len(seqs), L), self.n_channels, dtype=torch.long)
            for i, s in enumerate(seqs):
                X[i, : len(s)] = s
            n = len(seqs)
            for _ in range(epochs):
                perm = torch.randperm(n)
                for i in range(0, n, batch):
                    idx = perm[i : i + batch]
                    opt.zero_grad()
                    out = self.net(X[idx])
                    loss = lossf(out, y[idx])
                    loss.backward()
                    opt.step()
            return self

        def predict_proba_stream(self, obs_list):
            self.net.eval()
            with torch.no_grad():
                x = self._encode(obs_list).unsqueeze(0)
                logits = self.net(x)
                return torch.softmax(logits, dim=1).cpu().numpy()[0]
