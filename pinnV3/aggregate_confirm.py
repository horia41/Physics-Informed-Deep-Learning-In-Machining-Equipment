
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ["mae_overall_um", "mae_flank_wear_um",
           "mae_adhesion_um", "mae_flank_wear+adhesion_um"]
SHORT = {"mae_overall_um": "overall", "mae_flank_wear_um": "flank",
         "mae_adhesion_um": "adh", "mae_flank_wear+adhesion_um": "fa"}


def parse_runs(run_dir: Path) -> list[dict]:
    rows = []
    for rj in sorted(run_dir.glob("*/results.json")):
        d = json.loads(rj.read_text())
        name = d.get("experiment", rj.parent.name)
        phys = d.get("physics", {})
        cfg = d.get("config") or d.get("base_config") or {}
        lam = float(phys.get("lambda_max", 0.0) or 0.0)
        one = bool(phys.get("one_sided", False))
        tslope = bool(phys.get("use_taylor_slope", False))
        if lam == 0.0:
            variant = "ctrl"
        else:
            variant = ("ceil" if one else "sym") + ("-T" if tslope else "")
        m_s = re.search(r"_s(\d+)", name)
        seed = int(m_s.group(1)) if m_s else -1
        epochs = cfg.get("epochs")
        if epochs is None:
            m_e = re.search(r"_e(\d+)", name)
            epochs = int(m_e.group(1)) if m_e else -1
        for split, mm in d.get("splits", {}).items():
            rows.append(dict(family=run_dir.name, name=name, split=split,
                             variant=variant, lam=lam, epochs=int(epochs), seed=seed,
                             **{SHORT[k]: mm.get(k) for k in METRICS}))
    return rows


def show(df: pd.DataFrame, family: str, split: str = "test") -> None:
    sub = df[(df.family == family) & (df.split == split)]
    if sub.empty:
        return
    for ep in sorted(sub.epochs.unique()):
        cell = sub[sub.epochs == ep]
        agg = {f"{c}_m": (c, "mean") for c in SHORT.values()}
        agg.update({f"{c}_s": (c, "std") for c in SHORT.values()})
        g = cell.groupby(["variant", "lam"]).agg(n=("seed", "nunique"), **agg).reset_index()
        base = g[g.variant == "ctrl"]
        b = {c: (base[f"{c}_m"].iloc[0] if not base.empty else np.nan) for c in SHORT.values()}
        g = g.sort_values("overall_m")

        print(f"\n{'='*104}")
        print(f"  {family}  —  {split},  {int(ep)} epochs   (mean±std over seeds;  Δ vs same-epoch ctrl)")
        print('='*104)
        names = {"overall": "overall", "flank": "flank", "adh": "adhesion", "fa": "flank+adh"}
        head = f"  {'variant':8s} {'λ':>4} {'sd':>3}  " + "  ".join(f"{names[c]:>13}" for c in SHORT.values())
        head += "   " + "  ".join(f"Δ{names[c][:6]:>6}" for c in SHORT.values())
        print(head)
        for _, r in g.iterrows():
            cells = []
            for c in SHORT.values():
                sd = 0.0 if pd.isna(r[f"{c}_s"]) else r[f"{c}_s"]
                cells.append(f"{r[f'{c}_m']:6.2f}±{sd:4.2f}")
            deltas = [f"{r[f'{c}_m'] - b[c]:+6.2f}" if not np.isnan(b[c]) else "   -  "
                      for c in SHORT.values()]
            print(f"  {r.variant:8s} {r.lam:>4.2g} {int(r.n):>3}  " + "  ".join(cells)
                  + "   " + "  ".join(deltas))
        print("  (negative Δ = physics better than control at the same epoch count)")


def main():
    dirs = [Path(p) for p in sys.argv[1:]]
    if not dirs:
        sys.exit("Usage: python aggregate_confirm.py <run_dir> [<run_dir> ...]")
    rows = []
    for d in dirs:
        if d.exists():
            rows += parse_runs(d)
        else:
            print(f"[warn] {d} not found")
    if not rows:
        sys.exit("No results.json found.")
    df = pd.DataFrame(rows)
    for fam in df.family.unique():
        for split in ["test", "val"]:
            show(df, fam, split)
    out = dirs[0].parent / "confirm_summary.csv"
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}  ({df.name.nunique()} runs)")


if __name__ == "__main__":
    main()
