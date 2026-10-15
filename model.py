"""The transformer. 12 layers, width 200, 10 heads (LaBraM-Base size).

Input is a batch of windows. Each window is a set of channels; each channel is a set of 1-second
patches; each patch carries 8 product-E8 sub-tokens. A patch embedding is the sum of its 8
sub-token embeddings, plus an electrode-name embedding and a time-position embedding. The whole
window (channels x patches) is one sequence for the transformer. `features` returns a pooled
window representation (mean over real, non-padded positions).
"""
import torch
import torch.nn as nn

D = 200        # width
NL = 12        # layers
NH = 10        # heads
NB = 8         # sub-tokens per patch
NPATCH = 5     # patches per window the model was trained with (5 s)
SUB_V = 513    # sub-token vocabulary per block (512 cells + OOV)
N_ELEC = 101   # electrode vocabulary (100 tokens + padding)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub = nn.Embedding(NB * SUB_V, D)
        self.register_buffer("OFF", (torch.arange(NB) * SUB_V).view(1, 1, 1, NB))
        self.elec = nn.Embedding(N_ELEC, D)
        self.time = nn.Parameter(torch.zeros(1, 1, NPATCH, D))
        self.mask = nn.Parameter(torch.zeros(D))
        layer = nn.TransformerEncoderLayer(D, NH, 4 * D, batch_first=True, dropout=0.1, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, NL)
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, NB * SUB_V)   # masked-token prediction head (kept for reference)

    def _encode(self, tokens, elec, valid):
        # tokens: (B, C, P, 8) ids ; elec: (B, C) ids ; valid: (B, C) bool
        b, c, p, _ = tokens.shape
        h = self.sub(tokens + self.OFF).sum(-2)
        h = h + self.elec(elec)[:, :, None, :] + self.time
        h = h.reshape(b, c * p, D)
        pad = (~valid)[:, :, None].expand(b, c, p).reshape(b, c * p)
        return self.norm(self.tf(h, src_key_padding_mask=pad)), pad

    @torch.no_grad()
    def features(self, tokens, elec, valid):
        h, pad = self._encode(tokens, elec, valid)
        m = (~pad).float()[:, :, None]
        return (h * m).sum(1) / m.sum(1).clamp(min=1)   # (B, 200)


def load_model(path, device="cpu"):
    m = Model().to(device)
    m.load_state_dict(torch.load(path, map_location=device)["net"])
    m.eval()
    return m
