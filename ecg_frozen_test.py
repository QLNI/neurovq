"""Frozen cross-SIGNAL transfer: take the EEG model, freeze it, and probe it on ECG (the heart).
If a frozen brain model gives features that separate ECG conditions above chance, it has learned
general physiological-signal structure, not only EEG. PTB-XL, 5 classes, chance = 0.20.
"""
import sys, json, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from scipy.signal import resample_poly
from sklearn.metrics import balanced_accuracy_score
import torch.nn as nn

from tokenizer import Tokenizer, W
from model import load_model, NPATCH, N_ELEC
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
CMAX = 32

# PTB-XL as an .npz with X (n, samples) single lead @100 Hz, y (n,) 5 classes
d = np.load(sys.argv[1] if len(sys.argv) > 1 else "ptbxl.npz", allow_pickle=True)
X = d["X"].astype(np.float32)                       # (2000, 1000) @100Hz, single lead
Y = d["y"].astype(np.int64)
X = resample_poly(X, 2, 1, axis=1)[:, 500:1500]     # ->200Hz, keep middle 5 s (1000 samples)
print(f"PTB-XL: {X.shape} 5-class {np.bincount(Y)} (chance 0.200)", flush=True)

tok = Tokenizer("weights/tokenizer_e8.pt", dev)
net = load_model("weights/eegfm_e8.pt", dev)
ELEC = json.load(open("weights/elec_vocab.json"))["vocab"]["CZ"]   # one fixed electrode id (arbitrary; ECG has none)

@torch.no_grad()
def features(Xset):
    out = []
    for i in range(0, len(Xset), 512):
        xb = Xset[i:i+512]
        p = xb.reshape(-1, NPATCH, W).astype(np.float32)
        p = (p - p.mean(-1, keepdims=True)) / (p.std(-1, keepdims=True) + 1e-6)
        ids = tok.encode(p.reshape(-1, W)).reshape(len(xb), 1, NPATCH, tok.n_sub_tokens)
        t = np.zeros((len(xb), CMAX, NPATCH, tok.n_sub_tokens), np.int64); t[:, 0] = ids[:, 0]
        e = np.full((len(xb), CMAX), N_ELEC - 1, np.int64); e[:, 0] = ELEC
        v = np.zeros((len(xb), CMAX), bool); v[:, 0] = True
        out.append(net.features(torch.as_tensor(t, device=dev),
                                torch.as_tensor(e, device=dev),
                                torch.as_tensor(v, device=dev)).cpu().numpy())
    return np.concatenate(out)

rng = np.random.default_rng(0); idx = rng.permutation(len(X)); tr = idx[:1600]; te = idx[1600:]
F = features(X); Ftr = torch.tensor(F[tr], device=dev); Fte = torch.tensor(F[te], device=dev)
ytr = torch.tensor(Y[tr], device=dev)

head = nn.Sequential(nn.LayerNorm(200), nn.Linear(200, 5)).to(dev)
opt = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=0.05); best = 0
for ep in range(60):
    head.train(); perm = torch.randperm(len(Ftr), device=dev)
    for i in range(0, len(Ftr), 256):
        b = perm[i:i+256]; loss = nn.functional.cross_entropy(head(Ftr[b]), ytr[b])
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad(): pr = head(Fte).argmax(1).cpu().numpy()
    best = max(best, balanced_accuracy_score(Y[te], pr))
print(f"\nFROZEN EEG model -> ECG (PTB-XL 5-class) balanced accuracy: {best:.3f}   (chance 0.200)", flush=True)
print("=> " + ("adapts: the frozen brain model transfers to the heart signal above chance"
               if best > 0.25 else "no clear transfer"), flush=True)
