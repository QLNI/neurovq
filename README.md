# NeuroVQ

A compact EEG foundation model: a 6.5M-parameter transformer with a lattice
vector-quantization tokenizer. Trained from scratch on about 3,900 hours of public
EEG. The transformer is the same size as LaBraM-Base; the difference is the tokenizer.

## Results

Held-out test sets. Any patient appearing in a test split was removed from
pretraining, so these numbers are not inflated by leakage.

TUAB (abnormal-EEG detection), balanced accuracy, recording level:

    NeuroVQ, fine-tuned    0.837
    LaBraM (ICLR 2024)     0.814

Frozen backbone (transformer weights frozen, only a linear layer trained on top):

    TUAB (same domain as LaBraM's training data)
        NeuroVQ  0.751     LaBraM  0.784
    Sleep staging, 5 classes (a montage and domain LaBraM never trained on)
        NeuroVQ  0.393     LaBraM  0.200      chance = 0.200

On its own training domain LaBraM is strong frozen. On a new montage it drops to
chance; NeuroVQ keeps working, because the tokenizer is montage-agnostic.

## How it works

Tokenizer. A recording is cut into 1-second patches per channel (200 samples at
200 Hz). A small convolutional encoder maps each patch to 64 numbers. Those 64 are
split into 8 groups of 8, and each group is quantized to the nearest point of the E8
lattice (the optimal lattice quantizer in 8 dimensions; Conway-Sloane decoder). That
gives 8 discrete sub-tokens per patch. Because every E8 cell is reachable, the
codebook does not collapse: all cells stay in use.

Transformer. 12 layers, width 200, 10 heads. A patch embedding is the sum of its 8
sub-token embeddings, plus an embedding for the electrode name and one for the time
position. A window (channels by patches) is one sequence. Pretraining masks half the
patches and predicts their sub-tokens.

The electrode-name embedding is what lets one model take any montage: channels are
identified by name, not by position, so a 19-channel clinical recording and a
64-channel cap go through the same model.

## Codebook utilization (the point of the tokenizer)

A learned VQ codebook collapses as it grows: most entries die and are never selected.
The E8 lattice uses all of its cells. Measured on EEG patches (8-dimensional latent,
matched codebook size K):

    K        learned VQ used    E8 lattice used
    256      100%               100%
    1024      49%               100%
    4096      21%               100%
    8192      11%               100%

The lattice keeps 100% at every size; the learned codebook falls to about a tenth at
K = 8192. That is why a large token vocabulary can be used without wasting it, with no
codebook, no commitment loss, and no codebook-reset tricks.

## Files

    weights/eegfm_e8.pt        transformer weights (29 MB)
    weights/tokenizer_e8.pt    tokenizer encoder + lattice vocabulary (1.5 MB)
    weights/elec_vocab.json    electrode name -> embedding index
    e8.py                      E8 lattice decoder
    tokenizer.py               EEG patch -> tokens
    model.py                   transformer
    infer.py                   worked example

## Use

    pip install -r requirements.txt
    python infer.py                        # random 10 s example
    python infer.py eeg.npy F3 C3 P3 ...    # your array, one name per channel

Input is a NumPy array `(n_channels, n_samples)` at 200 Hz, in microvolts.
`featurize` returns one 200-dimensional vector per 5-second window. For a task,
train a linear layer on those vectors (backbone frozen), or fine-tune the whole
model.

## Input requirements

- 200 Hz. Resample first if your data is at another rate.
- Microvolts.
- Standard 10-20 / 10-10 electrode names (FP1, F3, Cz, ...). Old names T3/T4/T5/T6
  are mapped to T7/T8/P7/P8. Channels with unknown names are dropped.
- Bipolar montages: name the channel by the pair, e.g. FPZ-CZ.

## Notes

- The transformer was trained with 5-second windows (five 1-second patches) and up
  to 32 channels per window.
- The masked-prediction head is kept in the weights for reference; downstream use
  only needs the pooled features from `model.features`.

## Methods

Data curation, the tokenization pipeline, how benchmark leakage was avoided, and the
full set of tests are documented in [METHODS.md](METHODS.md).

## License

Non-commercial research use. See [LICENSE](LICENSE).
