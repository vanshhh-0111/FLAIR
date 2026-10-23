"""
Torch-free summary of benchmark_fewshot.py results (results.jsonl):
tables (markdown + LaTeX), paired statistics against baselines (bootstrap CI + exact sign-flip test over seeds),
ablations, effective-shot audit, and figures (mean +- std over seeds).

usage: python summarize_benchmark.py --results local_data/benchmark/results.jsonl --out_dir local_data/benchmark/summary
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRETTY = {"zero_shot": "Zero-shot FLAIR", "zero_shot_noknow": "Zero-shot (class names only)",
          "lp": "Linear probe", "clipAdapter": "CLIP-Adapter", "tipAdapter": "Tip-Adapter (repo defaults)",
          "tipAdapter-tuned": "Tip-Adapter (val-tuned)", "tipAdapter-f": "Tip-Adapter-F (repo defaults)",
          "tipAdapter-f-tuned": "Tip-Adapter-F (val-tuned alpha)", "ours": "Ours",
          "ours-nocache": "Ours w/o cache", "ours-noattn": "Ours w/o description attention",
          "ours-nores": "Ours w/o prototype residual", "ours-noknow": "Ours w/o expert text (class names)"}
DATASET_NAMES = {"02_MESSIDOR": "MESSIDOR (5-grade DR)", "13_FIVES": "FIVES (4-class)",
                 "25_REFUGE": "REFUGE (glaucoma)", "05_20x3": "20x3 (normal / RP / MH; tiny)"}
METRIC_NAMES = {"kappa": "Quadratic kappa", "aca": "Average class accuracy (ACA)", "f1": "Macro F1", "auc": "Macro AUC"}
PLOT_METHODS = ["ours", "clipAdapter", "tipAdapter-tuned", "tipAdapter-f-tuned"]
PLOT_COLORS = {"ours": "#2a78d6", "clipAdapter": "#eb6834", "tipAdapter-tuned": "#1baf7a",
               "tipAdapter-f-tuned": "#eda100"}
PAIRED_AGAINST = ["zero_shot", "lp", "clipAdapter", "tipAdapter", "tipAdapter-tuned", "tipAdapter-f",
                  "tipAdapter-f-tuned"]
ABLATIONS = ["ours-nocache", "ours-noattn", "ours-nores", "ours-noknow"]


def load(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        row = {"dataset": r["dataset"], "seed": r["seed"], "shots": r["shots"], "method": r["method"],
               "min_eff_shots": min(r["eff_shots"].values())}
        row.update(r["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def mean_std_table(df, dataset, metric, methods):
    sub = df[df.dataset == dataset]
    g = sub.groupby(["method", "shots"])[metric].agg(["mean", "std", "count"]).reset_index()
    shots = sorted(sub.shots.unique())
    out = {}
    for m in methods:
        out[m] = {}
        for k in shots:
            r = g[(g.method == m) & (g.shots == k)]
            out[m][k] = (float(r["mean"].iloc[0]), float(r["std"].iloc[0]) if len(r) else np.nan) if len(r) else None
    return out, shots


def write_tables(df, out_dir):
    methods = [m for m in PRETTY if m in set(df.method)]
    md, tex = [], []
    for metric in ["kappa", "aca", "f1"]:
        md.append(f"## {METRIC_NAMES[metric]} (mean ± std over seeds)\n")
        for ds in [d for d in DATASET_NAMES if d in set(df.dataset)]:
            tab, shots = mean_std_table(df, ds, metric, methods)
            best = {k: max((tab[m][k][0] for m in methods if tab[m][k]), default=None) for k in shots}
            md.append(f"**{DATASET_NAMES[ds]}**\n")
            md.append("| Method | " + " | ".join(f"{k}-shot" for k in shots) + " |")
            md.append("|---|" + "---|" * len(shots))
            tex.append(f"% {METRIC_NAMES[metric]} - {DATASET_NAMES[ds]}")
            tex.append("\\begin{table}[t]\\centering\\small")
            tex.append(f"\\caption{{{METRIC_NAMES[metric]} on {DATASET_NAMES[ds]} (mean $\\pm$ std over seeds).}}")
            tex.append("\\begin{tabular}{l" + "c" * len(shots) + "}\\toprule")
            tex.append("Method & " + " & ".join(f"{k}-shot" for k in shots) + " \\\\\\midrule")
            for m in methods:
                cells_md, cells_tex = [], []
                for k in shots:
                    v = tab[m][k]
                    if v is None:
                        cells_md.append("-"); cells_tex.append("-"); continue
                    s = f"{v[0]:.3f} ± {v[1]:.3f}"
                    st = f"{v[0]:.3f} $\\pm$ {v[1]:.3f}"
                    if abs(v[0] - best[k]) < 1e-12:
                        s, st = f"**{s}**", f"\\textbf{{{st}}}"
                    cells_md.append(s); cells_tex.append(st)
                md.append(f"| {PRETTY[m]} | " + " | ".join(cells_md) + " |")
                tex.append(f"{PRETTY[m]} & " + " & ".join(cells_tex) + " \\\\")
            tex.append("\\bottomrule\\end{tabular}\\end{table}\n")
            md.append("")
    open(os.path.join(out_dir, "tables.md"), "w").write("\n".join(md))
    open(os.path.join(out_dir, "tables.tex"), "w").write("\n".join(tex))


def sign_flip_p(d, rng, n_mc=20000):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    if len(d) <= 14:
        signs = np.array(list(itertools.product([-1, 1], repeat=len(d))))
        stats = np.abs((signs * d).mean(1))
    else:
        signs = rng.choice([-1, 1], size=(n_mc, len(d)))
        stats = np.abs((signs * d).mean(1))
    return float((stats >= obs - 1e-12).mean())


def per_seed_diff(df, a, b, metric, dataset=None):
    """Per-seed mean (over shots; and over datasets if dataset is None) of metric(a) - metric(b)."""
    sub = df if dataset is None else df[df.dataset == dataset]
    sub = sub[sub.method.isin([a, b])]
    piv = sub.pivot_table(index=["dataset", "shots", "seed"], columns="method", values=metric)
    if a not in piv or b not in piv:
        return None
    diff = (piv[a] - piv[b]).dropna().reset_index(name="d")
    per_ds_seed = diff.groupby(["dataset", "seed"])["d"].mean().reset_index()      # mean over shots
    per_seed = per_ds_seed.groupby("seed")["d"].mean()                              # equal weight per dataset
    return per_seed.values, diff


def paired_stats(df, target, others, metric, rng):
    rows = []
    for scope in [None] + [d for d in DATASET_NAMES if d in set(df.dataset)]:
        for o in others:
            res = per_seed_diff(df, target, o, metric, scope)
            if res is None:
                continue
            d, raw = res
            boots = rng.choice(d, size=(10000, len(d)), replace=True).mean(1)
            rows.append({"scope": "all datasets" if scope is None else DATASET_NAMES[scope],
                         "comparison": f"{PRETTY[target]} - {PRETTY[o]}", "metric": metric, "n_seeds": len(d),
                         "mean_diff": d.mean(), "ci95_lo": np.percentile(boots, 2.5),
                         "ci95_hi": np.percentile(boots, 97.5), "p_signflip": sign_flip_p(d, rng),
                         "win_rate_settings": float((raw["d"] > 0).mean())})
    return pd.DataFrame(rows)


def effective_shots_audit(df):
    g = df.groupby(["dataset", "shots"])["min_eff_shots"].min().unstack()
    lines = ["## Effective shots (smallest class support size actually available)\n",
             "Cells below the requested k mean at least one class ran out of support images.\n",
             "| Dataset | " + " | ".join(f"k={k}" for k in g.columns) + " |", "|---|" + "---|" * len(g.columns)]
    for ds in g.index:
        cells = [f"{int(v)}" + (" (capped)" if v < k else "") for k, v in g.loc[ds].items()]
        lines.append(f"| {DATASET_NAMES.get(ds, ds)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_metric(df, metric, out_path):
    surf, ink, sec, mut, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    dsets = [d for d in DATASET_NAMES if d in set(df.dataset)]
    n = len(dsets)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(5.4 * cols + 1.6, 3.7 * rows + 1.0), facecolor=surf, squeeze=False)
    for ax, ds in zip(axs.ravel(), dsets):
        ax.set_facecolor(surf)
        sub = df[df.dataset == ds]
        shots = sorted(sub.shots.unique())
        xs = np.arange(len(shots))
        zs = sub[sub.method == "zero_shot"][metric]
        if len(zs):
            ax.axhline(zs.mean(), color=mut, lw=1.2, ls="--", zorder=1)
            ax.text(xs[0], zs.mean(), " zero-shot", color=mut, fontsize=8, va="bottom")
        ends = {}
        for m in PLOT_METHODS:
            g = sub[sub.method == m].groupby("shots")[metric].agg(["mean", "std"]).reindex(shots)
            if g["mean"].isna().all():
                continue
            lw = 2.6 if m == "ours" else 1.8
            ax.plot(xs, g["mean"], color=PLOT_COLORS[m], lw=lw, marker="o", ms=5, mec=surf, mew=1.5, zorder=3)
            ax.fill_between(xs, g["mean"] - g["std"], g["mean"] + g["std"], color=PLOT_COLORS[m], alpha=0.12, lw=0)
            ends[m] = g["mean"].iloc[-1]
        # direct labels with simple collision nudging
        span = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1e-6)
        last = None
        for m in sorted(ends, key=lambda k: ends[k]):
            y = ends[m] if last is None or ends[m] - last >= 0.07 * span else last + 0.07 * span
            ax.annotate(PRETTY[m].replace(" (val-tuned alpha)", "").replace(" (val-tuned)", ""),
                        xy=(xs[-1], ends[m]), xytext=(xs[-1] + 0.12, y), color=sec, fontsize=8, va="center",
                        annotation_clip=False)
            last = y
        ax.set_xticks(xs)
        ax.set_xticklabels([str(k) for k in shots], color=mut, fontsize=9)
        ax.set_xlim(-0.2, len(shots) - 1 + 0.2)
        ax.tick_params(axis="y", colors=mut, labelsize=9, length=0)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(axis)
        ax.set_title(DATASET_NAMES[ds], loc="left", fontsize=10.5, color=ink, pad=8)
        ax.set_xlabel("shots per class", color=sec, fontsize=9)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{METRIC_NAMES[metric]} vs. shots per class (mean ± std over seeds)", x=0.04, ha="left",
                 fontsize=13, color=ink)
    handles = [plt.Line2D([0], [0], color=PLOT_COLORS[m], lw=2.2, marker="o", ms=5, mec=surf, label=PRETTY[m])
               for m in PLOT_METHODS if m in set(df.method)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
               labelcolor=sec)
    fig.tight_layout(rect=(0.01, 0.06, 0.86, 0.95))
    fig.savefig(out_path, dpi=200, facecolor=surf)
    plt.close(fig)


def df_to_md(d, floatfmt="{:.3f}"):
    cols = list(d.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                      for v in r.values) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="./local_data/benchmark/results.jsonl")
    p.add_argument("--out_dir", default="./local_data/benchmark/summary")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = load(args.results)
    df.to_csv(os.path.join(args.out_dir, "results_long.csv"), index=False)
    rng = np.random.default_rng(0)

    write_tables(df, args.out_dir)
    for metric in ["kappa", "aca"]:
        plot_metric(df, metric, os.path.join(args.out_dir, f"{metric}_vs_shots.png"))

    paired = pd.concat([paired_stats(df, "ours", PAIRED_AGAINST, m, rng) for m in ["kappa", "aca"]] +
                       [paired_stats(df, "ours", ABLATIONS, m, rng) for m in ["kappa", "aca"]], ignore_index=True)
    paired.to_csv(os.path.join(args.out_dir, "paired_stats.csv"), index=False)

    seeds = df.seed.nunique()
    report = ["# Few-shot FLAIR adapter benchmark - auto-generated summary\n",
              f"Runs: {len(df)}  |  datasets: {df.dataset.nunique()}  |  seeds: {seeds}  |  "
              f"methods: {df.method.nunique()}\n",
              effective_shots_audit(df),
              "## Paired comparisons (Ours minus other; per-seed averages, 95% bootstrap CI over seeds, "
              "exact sign-flip p-value)\n"]
    for metric in ["kappa", "aca"]:
        sub = paired[(paired.metric == metric) & (paired.scope == "all datasets")].drop(
            columns=["scope", "metric", "n_seeds"])
        report += [f"**{METRIC_NAMES[metric]}, all datasets pooled**\n", df_to_md(sub), ""]
    report += ["Per-dataset breakdown: see paired_stats.csv. Full tables: tables.md / tables.tex.\n"]
    open(os.path.join(args.out_dir, "report.md"), "w").write("\n".join(report))
    print(open(os.path.join(args.out_dir, "report.md")).read())


if __name__ == "__main__":
    main()
