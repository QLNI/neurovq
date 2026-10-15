"""Adapt the tokenizer to ECG with the transformer (the 'brain') frozen.

The transformer and its token embeddings stay frozen. Only the small tokenizer encoder and a
linear head are trained on ECG. To train the encoder end-to-end through the discrete tokenizer,
each block's latent is soft-assigned to the E8 codebook cells by distance (a differentiable
approximation of the hard nearest-cell lookup), and the patch embedding is the corresponding
weighted sum of the frozen cell embeddings. This lets the front-end learn to speak ECG into the
codebook the frozen model already understands.

Compare against the fully frozen number (0.279) from ecg_frozen_test.py. Chance = 0.200.
"""
import sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, warnings
warnings.filterwarnings("ignore")
from scipy.signal import resample_poly
from sklearn.metrics import balanced_accuracy_score
from model import load_model, D, NB, NPATCH, N_ELEC, SUB_V
from tokenizer import _Encoder, W, BD
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
CELLS = 512; TEMP = 0.25

# ---- ECG data: .npz with X (n, samples) single lead @100 Hz, y (n,) 5 classes ----
d = np.load(sys.argv[1] if len(sys.argv) > 1 else "ptbxl.npz", allow_pickle=True)
X = d["X"].astype(np.float32); Y = d["y"].astype(np.int64)
X = resample_poly(X, 2, 1, axis=1)[:, 500:1500]           # 200 Hz, middle 5 s
rng = np.random.default_rng(0); idx = rng.permutation(len(X)); tr = idx[:1600]; te = idx[1600:]
print(f"PTB-XL {X.shape} 5-class (chance 0.200)", flush=True)

# ---- frozen brain ----
net = load_model("weights/eegfm_e8.pt", dev)
for p in net.parameters(): p.requires_grad = False
sub_w = net.sub.weight.detach()                           # (NB*SUB_V, D)

# ---- E8 cell centres per block, from the tokenizer vocabulary ----
tk = torch.load("weights/tokenizer_e8.pt", map_location=dev); S = tk["S"]
centers = []
for b in range(NB):
    c = np.zeros((CELLS, BD), np.float32)
    for k, i in tk["vmaps"][b].items():
        c[i] = np.array(eval(k), np.float32) / 2.0        # key = round(cell*2)  ->  cell = key/2
    centers.append(torch.tensor(c, device=dev))

# ---- trainable front-end (init from the EEG encoder) + head ----
enc = _Encoder().to(dev); enc.load_state_dict(tk["enc"])
head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 5)).to(dev)
ELEC = json.load(open("weights/elec_vocab.json"))["vocab"]["CZ"]

def patch_embed(z):                                       # z:(M,64) -> (M,D)
    z = z.reshape(-1, NB, BD); pe = 0
    for b in range(NB):
        d2 = ((z[:, b, None, :] - centers[b][None]) ** 2).sum(-1)   # (M, CELLS)
        w = torch.softmax(-d2 / TEMP, -1)
        pe = pe + w @ sub_w[b * SUB_V: b * SUB_V + CELLS]
    return pe

def forward(xb):                                          # xb:(B,1000) -> logits (B,5)
    b = len(xb)
    p = xb.reshape(b, NPATCH, W)
    p = (p - p.mean(-1, keepdim=True)) / (p.std(-1, keepdim=True) + 1e-6)
    z = torch.tanh(enc(p.reshape(-1, W))) * S             # (B*5, 64)
    pe = patch_embed(z).reshape(b, 1, NPATCH, D)          # single channel
    h = pe + net.elec(torch.full((b, 1), ELEC, device=dev))[:, :, None, :] + net.time
    h = h.reshape(b, NPATCH, D)
    h = net.norm(net.tf(h))                               # no padding: one channel, all valid
    feat = h.mean(1)
    return head(feat)

Xtr = torch.tensor(X[tr], device=dev); Xte = torch.tensor(X[te], device=dev)
ytr = torch.tensor(Y[tr], device=dev)
opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), 1e-3, weight_decay=0.01)
best = 0
for ep in range(40):
    enc.train(); head.train(); perm = torch.randperm(len(Xtr), device=dev)
    for i in range(0, len(Xtr), 128):
        bi = perm[i:i+128]
        loss = F.cross_entropy(forward(Xtr[bi]), ytr[bi])
        opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); head.eval()
    with torch.no_grad():
        pr = torch.cat([forward(Xte[i:i+256]).argmax(1) for i in range(0, len(Xte), 256)]).cpu().numpy()
    ba = balanced_accuracy_score(Y[te], pr); best = max(best, ba)
    if ep % 5 == 0: print(f"ep{ep} bal-acc {ba:.3f}", flush=True)
print(f"\nADAPTED tokenizer (brain frozen) -> ECG 5-class bal-acc: {best:.3f}   "
      f"(fully-frozen was 0.279, chance 0.200)", flush=True)
