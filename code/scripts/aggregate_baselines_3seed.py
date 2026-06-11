
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["mae_overall_um", "mae_flank_wear_um",
           "mae_adhesion_um", "mae_flank_wear+adhesion_um"]
SHORT = {"mae_overall_um": "overall", "mae_flank_wear_um": "flank",
         "mae_adhesion_um": "adh", "mae_flank_wear+adhesion_um": "fa"}

# Published / earlier single-seed reference points, for context only.
REFERENCE = [
    ("paper_resnet50",          "test", 30.0, 14.0, 39.0, 91.0),
    ("vision_664_1seed (s42)",  "test", 19.0, 16.6, 37.2, 23.2),
    ("vision_647_1seed (s42)",  "test", 22.4, 20.3, 35.1, 27.1),
    ("t3_gated_md30_1seed(s42)","test", 21.44, 18.94, 40.13, 25.77),
]


def parse_runs(root: Path) -> pd.DataFrame:
    rows = []
    for rj in sorted(root.glob("*/s*/*/results.json")):
        config = rj.parent.name
        seed_dir = rj.parent.parent.name           # "s42"
        group = rj.parent.parent.parent.name       # "vision_664" / "fusion" ...
        m = re.search(r"s(\d+)", seed_dir)
        seed = int(m.group(1)) if m else -1
        try:
            d = pd.read_json(rj, typ="series")
        except ValueError:
            import json
            d = pd.Series(json.loads(rj.read_text()))
        splits = d.get("splits", {}) or {}
        for split, mm in splits.items():
            rows.append(dict(
                group=group, config=config, seed=seed, split=split,
                n=mm.get("n_total"),
                **{SHORT[k]: mm.get(k) for k in METRICS},
            ))
    if not rows:
        sys.exit(f"No results.json found under {root}/*/s*/*/")
    return pd.DataFrame(rows)


def show(df: pd.DataFrame, split: str) -> pd.DataFrame:
    sub = df[df.split == split]
    if sub.empty:
        return pd.DataFrame()
    agg = {f"{c}_m": (c, "mean") for c in SHORT.values()}
    agg.update({f"{c}_s": (c, "std") for c in SHORT.values()})
    g = (sub.groupby(["group", "config"])
            .agg(seeds=("seed", "nunique"), **agg)
            .reset_index()
            .sort_values(["group", "overall_m"]))

    print(f"\n{'='*94}\n  {split.upper()} — mean ± std over seeds\n{'='*94}")
    hdr = f"  {'group':12s} {'config':28s} {'n':>2s}  " \
          f"{'overall':>13s} {'flank':>13s} {'adh':>13s} {'fa':>13s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in g.iterrows():
        def cell(name):
            m, sd = r[f"{name}_m"], r[f"{name}_s"]
            sd = 0.0 if pd.isna(sd) else sd
            return f"{m:6.2f}±{sd:5.2f}"
        print(f"  {r.group:12s} {r.config:28s} {int(r.seeds):>2d}  "
              f"{cell('overall'):>13s} {cell('flank'):>13s} "
              f"{cell('adh'):>13s} {cell('fa'):>13s}")
    return g


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/baselines_3seed").resolve()
    print(f"Reading: {root}")
    df = parse_runs(root)

    # Per-seed detail (so individual runs are inspectable / sanity-checkable)
    print(f"\n{'='*94}\n  PER-SEED test overall MAE (µm)\n{'='*94}")
    piv = (df[df.split == "test"]
           .pivot_table(index=["group", "config"], columns="seed",
                        values="overall", aggfunc="first"))
    print(piv.round(2).to_string())

    g_test = show(df, "test")
    show(df, "val")

    print(f"\n{'-'*94}\n  Reference points (earlier single-seed / paper, test split):")
    for name, _split, ov, fl, ad, fa in REFERENCE:
        print(f"    {name:26s} overall={ov:6.2f}  flank={fl:6.2f}  "
              f"adh={ad:6.2f}  fa={fa:6.2f}")

    out_csv = root / "baselines_3seed_summary.csv"
    if not g_test.empty:
        g_test.to_csv(out_csv, index=False)
        print(f"\n  Test summary → {out_csv}")


if __name__ == "__main__":
    main()
