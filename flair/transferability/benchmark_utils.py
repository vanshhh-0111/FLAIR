"""
Torch-free utilities for the cached-feature few-shot benchmark (benchmark_fewshot.py).

The split logic mirrors flair.transferability.data.dataloader.get_dataloader_splits (same seeded shuffle,
class-wise partition, nested k-shot support sets) but adds a held-out validation split that is carved out of
the data NOT used for testing or for the support pool, so hyper-parameters can be tuned without touching the
test set. Because image features are extracted once, splits are plain index operations.
"""

import random
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Hyper-parameter grids (tuned on the validation split only)
ALPHAS = [0, 1, 5, 10, 20, 50, 100, 200]      # weight of the Tip-style cache term (0 = no cache)
BETAS = [1, 3, 5, 10]                          # sharpness of the Tip-style cache affinity
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0]              # L2 penalty on the prototype residual (ours)


def make_split(labels, seed, test_frac=0.5, val_frac=0.1):
    """
    Class-wise split. For every class, in the repo's seeded-shuffle order:
      first test_frac        -> test
      last  val_frac         -> validation (used only for hyper-parameter selection)
      the rest (in order)    -> support pool; the k-shot support set is the first k pool items of each class,
                                so support sets are nested (1-shot subset of 2-shot subset of ...).
    With val_frac=0 and test_frac=0.8 this reproduces the repo's own train/test partition.
    """
    labels = np.asarray(labels)
    order = list(range(len(labels)))
    random.Random(seed).shuffle(order)          # same permutation as random.seed(seed); random.shuffle(list)
    order = np.asarray(order)

    test, val, pool = [], [], {}
    for c in np.unique(labels):
        idx = order[labels[order] == c].tolist()
        n_test = int(test_frac * len(idx))
        n_val = int(val_frac * len(idx))
        test += idx[:n_test]
        rest = idx[n_test:]
        n_train = len(rest) - n_val
        pool[int(c)] = rest[:n_train]
        val += rest[n_train:]
    return {"test": np.asarray(test, dtype=int), "val": np.asarray(val, dtype=int), "pool": pool}


def kshot_indices(pool, k):
    """First k support items of every class. Returns (indices, effective shots per class)."""
    idx, effective = [], {}
    for c in sorted(pool):
        take = pool[c][:k]
        idx += take
        effective[int(c)] = len(take)
    return np.asarray(idx, dtype=int), effective


def quick_metrics(refs, preds):
    """Robust ACA (balanced accuracy) and quadratic kappa; safe on tiny validation sets."""
    refs = np.asarray(refs)
    pred = np.asarray(preds).argmax(-1)
    aca = float(np.mean([(pred[refs == c] == c).mean() for c in np.unique(refs)]))
    try:
        kappa = float(cohen_kappa_score(refs, pred, weights="quadratic"))
    except Exception:
        kappa = float("nan")
    if not np.isfinite(kappa):
        kappa = 0.0
    return {"aca": aca, "kappa": kappa}


def tuning_score(refs, preds):
    """Validation criterion used to pick hyper-parameters: mean of ACA and quadratic kappa."""
    m = quick_metrics(refs, preds)
    return 0.5 * (m["aca"] + m["kappa"])
