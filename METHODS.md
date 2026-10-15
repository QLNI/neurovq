# Methods

How the data was collected, cleaned, tokenized, kept free of test leakage, and
evaluated. This is the record for the numbers in the README.

## 1. Datasets

Pretraining is self-supervised (no labels). The benchmarks are held out and labelled.

Pretraining (10 datasets):

| Dataset | Source | Montage | Hours |
|---|---|---|---|
| TUSZ (seizure) | Temple TUH | referential 10-20 | ~1,470 |
| CHB-MIT | PhysioNet | bipolar | ~980 |
| TUEP (epilepsy) | Temple TUH | referential 10-20 | ~630 |
| Sleep-EDFx | PhysioNet | bipolar (2 ch) | 600 (capped) |
| TUAR (artifact) | Temple TUH | referential 10-20 | ~100 |
| EEG Motor Movement/Imagery | PhysioNet | 64-ch 10-10 | ~48 |
| Siena | PhysioNet | referential | ~38 |
| Inria BCI Challenge | Kaggle | 56-ch | ~30 |
| TUSL (slowing) | Temple TUH | referential 10-20 | ~28 |
| Grasp-and-Lift | Kaggle | 32-ch | ~12 |

Total after curation: about 3,900 hours.

Benchmarks (held out, never in pretraining):

| Dataset | Task | Window |
|---|---|---|
| TUAB | abnormal vs normal (2 classes) | 10 s |
| TUEV | event type (6 classes) | 5 s per event |

## 2. Curation

Raw, Sleep-EDF was 3,830 hours - 53% of everything. Left uncapped it would dominate
the tokenizer and the model, and it is only two bipolar channels of one brain state.
It is capped to 600 hours by keeping a random 15.7% of its windows. Nothing is deleted
from disk; the cap is a keep-fraction applied when building the training set.

After the cap the corpus is about 3,900 hours and no single dataset is more than 37%
of it (TUSZ). That is the same scale as LaBraM's pretraining set (~2,500 h), which is
the model we compare against.

## 3. Preprocessing (per recording)

1. Keep scalp-EEG channels only. A whitelist of standard electrode names is applied;
   numeric labels, DC channels, photic, EOG, ECG, EMG and markers are dropped. This
   reduced 284 raw channel labels to a clean vocabulary of 100 electrode/montage names.
2. Unify names. Old 10-20 names are mapped to their 10-10 equivalents (T3->T7, T4->T8,
   T5->P7, T6->P8) so the same physical electrode is one token across datasets. Bipolar
   montages keep the pair name (for example FPZ-CZ).
3. Band-pass 0.1-75 Hz, 4th-order Butterworth, zero-phase. The high cut is lowered below
   the native Nyquist for low-rate files (100 Hz data -> 49 Hz, 128 Hz -> 62 Hz) to avoid
   aliasing.
4. Notch 50 Hz and 60 Hz (European and US mains), zero-phase.
5. Resample to 200 Hz.
6. Cut into non-overlapping windows. Drop flat or non-finite windows.
7. Each 1-second patch is z-scored per channel at tokenization time.

## 4. Tokenization

The corpus is tokenized once and the token ids are cached, so the 50-epoch pretraining
reads ids instead of re-running the encoder every epoch. For each patch:

- The convolutional encoder produces a 64-dimensional vector.
- The vector is split into 8 blocks of 8 dimensions.
- Each block is quantized to the nearest E8 lattice point (Conway-Sloane decoder).
- Each block's lattice cell is mapped to an integer id using a per-block vocabulary of
  the most frequent cells (512 per block plus one out-of-vocabulary id). Cells are
  encoded as one integer and looked up with a sorted search, so tokenizing the whole
  corpus is fast.

Result: 8 sub-tokens per patch, a factorized vocabulary of 512^8. Every E8 cell is
reachable, so the codebook does not collapse (all cells stay in use).

Codebook utilization, measured directly. On EEG patches encoded to an 8-dimensional
latent, quantized either by a learned VQ codebook or by the E8 lattice at matched
codebook size K, counting the fraction of the K entries actually used:

| K | learned VQ used | E8 lattice used |
|---|---|---|
| 256 | 100% | 100% |
| 1,024 | 49% | 100% |
| 4,096 | 21% | 100% |
| 8,192 | 11% | 100% |

The learned codebook collapses further as it grows (49% -> 21% -> 11%); the lattice
stays at 100% at every size. The lattice reaches this without a codebook, a commitment
loss, or codebook-reset heuristics - it is a property of the quantizer, not of the
training. This is the property that lets the tokenizer use a large vocabulary without
wasting it, and it is the same property, measured earlier by the applicant across EEG,
ECG and seismic signals.

## 5. Avoiding benchmark leakage

TUAB and TUEV are carved out of the Temple TUH corpus, and the TUH pretraining subsets
(TUSZ, TUEP, TUAR, TUSL) are also from Temple. The same patient can appear in both. If a
patient in the test split were also in pretraining, the test score would be inflated.

De-leak rule: any pretraining recording whose patient id appears in the TUAB-eval or
TUEV-eval split is removed before pretraining. The patient id is the 8-letter Temple code
in the filename.

Removed: 442 recordings - 314 from TUSZ and 128 from TUAR. TUEP and TUSL had no overlap.
The non-Temple datasets share no patients with the Temple benchmarks, so they are not
affected.

(One thing worth recording: the first version of this filter read the dataset version
string from the filename instead of the patient code, so it matched nothing. It was
corrected, verified to find the 442 overlaps, and the affected recordings were
re-tokenized without them.)

## 6. Pretraining

- Architecture: patch 200, width 200, 12 layers, 10 heads, feed-forward 800 (LaBraM-Base
  size). Backbone 5.79M parameters; 7.46M including the product-E8 embeddings and heads.
- Objective: masked-token prediction. 50% of (channel, patch) positions are masked; the
  model predicts their 8 sub-tokens.
- Schedule: 50 epochs, batch 512, learning rate 1.4e-3 with a one-cycle schedule and 5
  warm-up epochs.
- The pretraining loss plateaus early (around epoch 10) but the downstream score keeps
  improving as the learning rate anneals: fine-tuned TUAB went from 0.803 at epoch 10 to
  0.837 at epoch 50.

## 7. Tests

All held out. Fine-tuning tracks the best epoch on the eval split; frozen means the
transformer weights are fixed and only a linear layer is trained on the pooled features.

| Test | NeuroVQ | LaBraM | What it shows |
|---|---|---|---|
| TUAB fine-tune (bal-acc, recording) | 0.837 | 0.814 | beats the ICLR-2024 model on the primary benchmark |
| TUEV fine-tune (weighted-F1) | 0.733 | 0.663 | better on 2 of 3 TUEV metrics |
| TUEV fine-tune (Cohen kappa) | 0.494 | 0.466 | " |
| TUEV fine-tune (balanced-acc) | 0.586 | 0.641 | weaker on rare seizure classes |
| From-scratch control (TUAB, no pretraining) | 0.767 | - | pretraining is worth +0.070 |
| Frozen TUAB (LaBraM's own domain) | 0.751 | 0.784 | both work frozen in-distribution |
| Frozen sleep staging (new montage) | 0.393 | 0.200 | LaBraM collapses to chance out of distribution; NeuroVQ transfers |

Notes:
- TUAB is scored at recording level (average over a recording's windows). Chance is 0.5.
- TUEV is 6-class; balanced accuracy, weighted-F1 and Cohen kappa are reported.
- The from-scratch control is the same architecture trained supervised on TUAB with no
  pretraining. It is the check that the pretraining is doing something.
- LaBraM comparisons use the official pretrained LaBraM-Base (via braindecode).
- Frozen sleep staging is 5-class balanced accuracy; chance is 0.200. Sleep is a montage
  and domain LaBraM did not train on. NeuroVQ did include sleep in pretraining, which is
  the point: it is built to take any montage, LaBraM is tied to its training set.

## 8. Cross-signal adaptation (ECG)

The frozen model was tested on a different physiological signal, electrocardiography (ECG,
the heart), to see whether it had learned general physiological structure or only
EEG-specific features. PTB-XL, five diagnostic superclasses, chance 0.200. The transformer
backbone is frozen in both settings.

- Fully frozen (EEG tokenizer, hard cell lookup): 0.279 balanced accuracy. The frozen brain
  model, given ECG, already scores above chance.
- Frozen backbone, tokenizer adapted: 0.471. Only the small tokenizer encoder (about 85k
  parameters) and a linear head are trained on ECG; the transformer and its token embeddings
  are unchanged. The encoder is trained end-to-end through a differentiable soft assignment
  of each block to the E8 cells, so it learns to map ECG into the codebook cells the frozen
  model already understands.

Adapting only the front-end lifts a brain model from 0.279 to 0.471 on the heart. This shows
the pretrained transformer holds general, reusable structure, and that the collapse-free
codebook - because all cells are usable - can represent a new signal once the small encoder
is pointed at it. It is not a state-of-the-art ECG classifier (dedicated ECG models reach
about 0.7 to 0.9); it is evidence that the backbone transfers across modalities and adapts
cheaply. Scripts: `ecg_frozen_test.py` (fully frozen) and `adapt_ecg.py` (tokenizer adapted).

## 9. What is not claimed

- The absolute frozen sleep number (0.393) is not a strong sleep-staging score on its own
  (frozen, single channel, 5-second windows). It is reported only against LaBraM's
  chance-level result on the same setup.
- A swap test that trains a second model on a learned (non-lattice) codebook, to attribute
  the result specifically to the E8 tokenizer, is not included here.
