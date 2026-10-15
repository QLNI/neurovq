"""Product-E8 tokenizer: turns a 1-second EEG patch into 8 discrete sub-tokens.

Each patch (200 samples) is encoded to a 64-dim vector, split into 8 blocks of 8 dims, and each
block is quantized to the nearest E8 lattice point. Lattice cells are mapped to integer ids using
the per-block vocabularies stored in the weight file. Cells not seen at fit time map to an OOV id.
"""
import json
import numpy as np
import torch
import torch.nn as nn

from e8 import decode_e8

W = 200        # samples per patch (1 s at 200 Hz)
DIM = 64       # latent width
NB = 8         # number of E8 blocks
BD = 8         # dims per block


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, 2, 3), nn.GELU(),
            nn.Conv1d(32, 64, 7, 2, 3), nn.GELU(),
            nn.Conv1d(64, 64, 7, 2, 3), nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 25, 128), nn.GELU(),
            nn.Linear(128, DIM),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))


class Tokenizer:
    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location=device)
        self.device = device
        self.scale = ck["S"]
        self.enc = _Encoder().to(device)
        self.enc.load_state_dict(ck["enc"])
        self.enc.eval()
        # per-block lookup: key (8 ints) -> id, encoded as one integer for fast search
        base, off = 32, 16
        self.base_w = (base ** np.arange(BD)).astype(np.int64)
        self.keys, self.ids, self.oov = [], [], []
        for vm in ck["vmaps"]:
            k = np.array([self._key(np.array(eval(s))) for s in vm.keys()], np.int64)
            v = np.array(list(vm.values()), np.int64)
            order = np.argsort(k)
            self.keys.append(k[order])
            self.ids.append(v[order])
            self.oov.append(len(vm))
        self.n_sub_tokens = NB
        self.vocab_per_block = self.oov[0]

    def _key(self, a):
        return ((a.astype(np.int64) + 16) * self.base_w).sum(-1)

    @torch.no_grad()
    def encode(self, patches):
        """patches: (M, 200) float32, z-scored per patch. Returns ids (M, 8) int64."""
        x = torch.as_tensor(patches, dtype=torch.float32, device=self.device)
        z = (torch.tanh(self.enc(x)) * self.scale).reshape(-1, NB, BD)
        cells = torch.round(decode_e8(z.reshape(-1, BD)) * 2).long().reshape(-1, NB, BD).cpu().numpy()
        keyed = ((cells + 16) * self.base_w).sum(-1)     # (M, NB)
        out = np.empty((len(x), NB), np.int64)
        for b in range(NB):
            k, ids, oov = self.keys[b], self.ids[b], self.oov[b]
            pos = np.clip(np.searchsorted(k, keyed[:, b]), 0, len(k) - 1)
            out[:, b] = np.where(k[pos] == keyed[:, b], ids[pos], oov)
        return out


def to_patches(signal):
    """signal: (n_channels, n_samples) at 200 Hz. Returns per-channel 1s patches, z-scored,
    as (n_channels, n_patches, 200) float32."""
    signal = np.asarray(signal, np.float32)
    c, t = signal.shape
    n = t // W
    p = signal[:, :n * W].reshape(c, n, W)
    p = (p - p.mean(-1, keepdims=True)) / (p.std(-1, keepdims=True) + 1e-6)
    return p
