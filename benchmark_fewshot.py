"""
Cached-feature few-shot benchmark for FLAIR adapters.

1) Extract FLAIR image features ONCE per dataset (no augmentation) and cache them as small .npz files
   (re-used across sessions: put --feature_dir on Google Drive).
2) For every dataset x seed x k-shot, build a class-wise split (flair.transferability.benchmark_utils.make_split:
   held-out test, held-out validation for hyper-parameter selection, nested k-shot support sets) and run every
   method on the cached features. The repo's own adapter classes are re-used unchanged (only their feature
   extraction is replaced by a lookup), so baselines are the repo's baselines.
3) Every finished run is appended to results.jsonl immediately (resumable, disconnect-safe).

Hyper-parameters of the *_tuned / ours variants are chosen on the validation split only, never on test data.
Summaries/plots/statistics: summarize_benchmark.py (torch-free).
"""

import argparse
import contextlib
import io
import json
import os
import time
import types

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from flair.modeling.model import FLAIRModel
from flair.modeling.misc import set_seeds
from flair.pretraining.data.dataset import Dataset
from flair.pretraining.data.transforms import LoadImage, ImageScaling, CopyDict
from flair.transferability.modeling import adapters as A
from flair.transferability.benchmark_utils import (make_split, kshot_indices, tuning_score,
                                                   ALPHAS, BETAS, LAMBDAS)
from flair.utils.metrics import classification_metrics

from local_data.constants import PATH_DATASETS
from local_data.experiments import get_experiment_setting

import warnings
warnings.filterwarnings("ignore")

ALL_METHODS = ["zero_shot", "zero_shot_noknow", "lp", "clipAdapter", "tipAdapter", "tipAdapter-tuned",
               "tipAdapter-f", "tipAdapter-f-tuned", "ours", "ours-nocache", "ours-noattn", "ours-nores",
               "ours-noknow"]


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ----------------------------------------------------------------------------------------------------------
# Feature extraction (once per dataset)
# ----------------------------------------------------------------------------------------------------------
def load_or_extract_features(model, dataset, data_root, feature_dir, batch_size=8, num_workers=2, size=(512, 512)):
    path = os.path.join(feature_dir, f"features_{dataset}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        print(f"[features] loaded {path}: {z['X'].shape}")
        return z["X"].astype(np.float32), z["Y"].astype(np.int64), z["paths"]

    setting = get_experiment_setting(dataset)
    df = pd.read_csv(setting["dataframe"])
    data = [{"image_path": data_root + df.loc[i, "image"],
             "label": setting["targets"][eval(df.loc[i, "categories"])[0]]} for i in range(len(df))]
    transforms = Compose([CopyDict(), LoadImage(crop_foreground=True), ImageScaling(size=size)])
    loader = DataLoader(Dataset(data=data, transform=transforms), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, drop_last=False)

    base = A.AdapterWrapper(model, setting["targets"])         # re-use the repo's feature extraction
    t0 = time.time()
    X, Y = base.extract_vision_features(loader)
    paths = np.array([d["image_path"] for d in data])
    os.makedirs(feature_dir, exist_ok=True)
    np.savez(path, X=X.astype(np.float32), Y=Y.astype(np.int64), paths=paths)
    print(f"[features] extracted {dataset}: {X.shape} in {(time.time() - t0) / 60:.1f} min -> {path}")
    return X.astype(np.float32), Y.astype(np.int64), paths


# ----------------------------------------------------------------------------------------------------------
# Running adapters on cached features
# ----------------------------------------------------------------------------------------------------------
def attach(adapter, store):
    """Replace the adapter's image->feature extraction by a lookup into cached features."""
    def extract(loader, transforms=None):          # transforms (e.g. Tip-Adapter's test-time aug.) ignored
        X, Y = store[loader]
        return X.copy(), Y.copy()
    adapter.extract_vision_features = extract
    return adapter


def build(cls, model, targets, store, domain_knowledge=True, **kw):
    with quiet():
        adapter = cls(model, targets, domain_knowledge=domain_knowledge, **kw)
    return attach(adapter, store)


def fit(adapter):
    with quiet():
        adapter.fit({"train": "train"})


def test_metrics(refs, preds):
    try:
        m = classification_metrics(refs, preds)
        return {"aca": float(m["aca"]), "kappa": float(m["kappa"]), "f1": float(m["f1_avg"]),
                "auc": float(m["auc_avg"])}
    except Exception as e:                           # e.g. a class absent from a tiny split
        print("  [warn] full metrics failed:", e)
        return {"aca": float("nan"), "kappa": float("nan"), "f1": float("nan"), "auc": float("nan")}


def tune_cache(adapter, alphas, betas):
    """Pick (alpha, beta) on the validation split; ties -> first in grid order (smaller alpha)."""
    best = (-np.inf, adapter.alpha, adapter.beta)
    for alpha in alphas:
        for beta in betas:
            adapter.alpha, adapter.beta = alpha, beta
            refs, preds = adapter.predict("val")
            s = tuning_score(refs, preds)
            if s > best[0]:
                best = (s, alpha, beta)
    adapter.alpha, adapter.beta = best[1], best[2]
    return {"alpha": best[1], "beta": best[2], "val_score": float(best[0])}


def run_method(method, model, targets, store, seed):
    set_seeds(1000 + seed, use_cuda=torch.cuda.is_available())
    hp = {}

    if method in ("zero_shot", "zero_shot_noknow"):
        ad = build(A.ZeroShot, model, targets, store, domain_knowledge=(method == "zero_shot"))
    elif method == "lp":
        clf = LogisticRegression(random_state=0, C=0.316, max_iter=1000, class_weight="balanced")
        clf.fit(*store["train"])
        refs = store["test"][1]
        return test_metrics(refs, clf.predict_proba(store["test"][0])), hp
    elif method == "clipAdapter":
        ad = build(A.ClipAdapter, model, targets, store)
        fit(ad)
    elif method in ("tipAdapter", "tipAdapter-tuned"):
        ad = build(A.TipAdapter, model, targets, store, train=False)
        fit(ad)
        if method == "tipAdapter-tuned":
            hp = tune_cache(ad, ALPHAS, BETAS)
    elif method in ("tipAdapter-f", "tipAdapter-f-tuned"):
        ad = build(A.TipAdapter, model, targets, store, train=True)
        fit(ad)
        if method == "tipAdapter-f-tuned":
            hp = tune_cache(ad, ALPHAS, [5])           # beta is baked into F's training (5); tune alpha only
    elif method.startswith("ours"):
        use_attn = method != "ours-noattn"
        use_res = method != "ours-nores"
        alphas = [0] if method == "ours-nocache" else ALPHAS
        lambdas = LAMBDAS if use_res else [0.0]
        dk = method != "ours-noknow"
        best = None
        for lam in lambdas:
            cand = build(A.KnowledgeProtoAdapter, model, targets, store, domain_knowledge=dk,
                         use_attn=use_attn, use_res=use_res, lam=lam)
            fit(cand)
            info = tune_cache(cand, alphas, BETAS if method != "ours-nocache" else [5])
            if best is None or info["val_score"] > best[0]["val_score"]:
                best = (info, lam, cand)
        info, lam, ad = best
        hp = dict(info, lam=lam)
    else:
        raise ValueError(method)

    with torch.no_grad():
        refs, preds = ad.predict("test")
    return test_metrics(refs, preds), hp


# ----------------------------------------------------------------------------------------------------------
def parse_list(s, cast=str):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part and cast is int and part.count("-") == 1 and not part.startswith("-"):
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(cast(part))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="02_MESSIDOR,13_FIVES,25_REFUGE,05_20x3")
    p.add_argument("--methods", default=",".join(ALL_METHODS))
    p.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 0,1,2")
    p.add_argument("--shots", default="1,2,4,8,16")
    p.add_argument("--test_frac", type=float, default=0.5)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--data_root_path", default=PATH_DATASETS)
    p.add_argument("--feature_dir", default="./local_data/features")
    p.add_argument("--out_dir", default="./local_data/benchmark")
    p.add_argument("--extract_only", action="store_true")
    args = p.parse_args()

    datasets = parse_list(args.datasets)
    methods = parse_list(args.methods)
    seeds = parse_list(args.seeds, int)
    shots_list = parse_list(args.shots, int)
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "results.jsonl")

    done = set()
    if os.path.exists(results_path):
        for line in open(results_path):
            r = json.loads(line)
            done.add((r["dataset"], r["seed"], r["shots"], r["method"]))
    print(f"[resume] {len(done)} finished runs found in {results_path}")

    model = FLAIRModel.from_pretrained("jusiro2/FLAIR")
    model.project_features = True
    model.norm_features = True
    A.copy = types.SimpleNamespace(deepcopy=lambda m: m)   # share the (frozen) model across adapters: faster

    split_info = {}
    for dataset in datasets:
        X, Y, _ = load_or_extract_features(model, dataset, args.data_root_path, args.feature_dir)
        if args.extract_only:
            continue
        targets = get_experiment_setting(dataset)["targets"]
        t_ds = time.time()
        for seed in seeds:
            sp = make_split(Y, seed, args.test_frac, args.val_frac)
            counts = lambda idx: {int(c): int((Y[idx] == c).sum()) for c in np.unique(Y)}
            split_info[f"{dataset}/seed{seed}"] = {"test": counts(sp["test"]), "val": counts(sp["val"]),
                                                   "pool": {c: len(v) for c, v in sp["pool"].items()}}
            Xte, Yte = X[sp["test"]], Y[sp["test"]]
            Xva, Yva = X[sp["val"]], Y[sp["val"]]
            for k in shots_list:
                tr, eff = kshot_indices(sp["pool"], k)
                store = {"train": (X[tr], Y[tr]), "val": (Xva, Yva), "test": (Xte, Yte)}
                for method in methods:
                    if (dataset, seed, k, method) in done:
                        continue
                    t0 = time.time()
                    metrics, hp = run_method(method, model, targets, store, seed)
                    rec = {"dataset": dataset, "seed": seed, "shots": k, "eff_shots": eff, "method": method,
                           "metrics": metrics, "hp": hp}
                    with open(results_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"{dataset} seed{seed} k={k:<2d} {method:<20s} aca={metrics['aca']:.3f} "
                          f"kappa={metrics['kappa']:.3f} f1={metrics['f1']:.3f} ({time.time() - t0:.1f}s)")
        print(f"[done] {dataset} in {(time.time() - t_ds) / 60:.1f} min")

    if split_info:
        with open(os.path.join(args.out_dir, "split_info.json"), "w") as f:
            json.dump(split_info, f, indent=1)


if __name__ == "__main__":
    main()
