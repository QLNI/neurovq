"""Run the model on an EEG window and get a feature vector.

Input: a NumPy array (n_channels, n_samples) sampled at 200 Hz in microvolts, plus the electrode
name of each channel. Output: one 200-dim feature vector per 5-second window. Attach your own
linear layer on top of these features for a downstream task (see the frozen-probe results in the
README).

    python infer.py                 # runs on a random example
    python infer.py my_eeg.npy F3 C3 P3 ...   # your array + channel names
"""
import sys
import json
import numpy as np
import torch

from tokenizer import Tokenizer, to_patches, W
from model import load_model, NPATCH, N_ELEC

CMAX = 32          # max channels per window the model was trained with
SR = 200           # required sample rate (Hz)

_VOCAB = json.load(open("weights/elec_vocab.json"))["vocab"]
_OLD2NEW = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def electrode_id(name):
    """Map a channel name to an electrode-embedding index, or None if unknown."""
    s = name.upper().replace("EEG ", "").replace("-REF", "").replace("-LE", "").replace(".", "").strip()
    s = _OLD2NEW.get(s, s)
    return _VOCAB.get(s, None)


def featurize(signal, ch_names, tokenizer, model, device="cpu"):
    """signal: (n_channels, n_samples) @200 Hz uV. ch_names: list[str].
    Returns features (n_windows, 200)."""
    keep = [(i, electrode_id(n)) for i, n in enumerate(ch_names) if electrode_id(n) is not None]
    if not keep:
        raise ValueError("no recognised electrode names")
    rows = [i for i, _ in keep][:CMAX]
    elec_ids = [e for _, e in keep][:CMAX]
    patches = to_patches(signal[rows])                     # (C, n_patch, 200)
    c, n_patch, _ = patches.shape
    n_win = n_patch // NPATCH
    if n_win == 0:
        raise ValueError("signal shorter than one 5-second window")

    ids = tokenizer.encode(patches.reshape(-1, W)).reshape(c, n_patch, tokenizer.n_sub_tokens)
    feats = []
    for w in range(n_win):
        sub = ids[:, w * NPATCH:(w + 1) * NPATCH, :]        # (C, 5, 8)
        tok = np.zeros((1, CMAX, NPATCH, tokenizer.n_sub_tokens), np.int64)
        el = np.full((1, CMAX), N_ELEC - 1, np.int64)
        val = np.zeros((1, CMAX), bool)
        tok[0, :c] = sub
        el[0, :c] = elec_ids
        val[0, :c] = True
        f = model.features(torch.as_tensor(tok, device=device),
                           torch.as_tensor(el, device=device),
                           torch.as_tensor(val, device=device))
        feats.append(f.cpu().numpy())
    return np.concatenate(feats)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer("weights/tokenizer_e8.pt", dev)
    net = load_model("weights/eegfm_e8.pt", dev)

    if len(sys.argv) > 1:
        sig = np.load(sys.argv[1]).astype(np.float32)
        names = sys.argv[2:]
        if len(names) != sig.shape[0]:
            raise SystemExit("give one channel name per row of the array")
    else:
        names = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
                 "F7", "F8", "T7", "T8", "P7", "P8", "FZ", "CZ", "PZ"]
        sig = (np.random.randn(len(names), SR * 10) * 20).astype(np.float32)   # 10 s of noise
        print("no input given -> running on random 10 s / 19 channels")

    feats = featurize(sig, names, tok, net, dev)
    print("features:", feats.shape, "(n_windows, 200)")
    print("first vector (first 8 dims):", np.round(feats[0, :8], 3))
