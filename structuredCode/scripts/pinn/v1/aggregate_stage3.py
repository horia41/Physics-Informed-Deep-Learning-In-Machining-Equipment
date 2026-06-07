"""
Aggregate Stage-3 Taylor results into before/after tables.

Reads every results.json under each given run directory and prints, per family,
a table sorted by test overall MAE, with deltas vs that family's control
(the run whose name contains 'ctrl'). Also writes stage3_summary.csv.

    python aggregate_stage3.py \
        /scratch-shared/hionescu/project_pinn/pinn/runs/stage3_vision \
        /scratch-shared/hionescu/project_pinn/pinn/runs/stage3_fusion
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import pandas as pd

METRICS = ["mae_overall_um", "mae_flank_wear_um",
           "mae_adhesion_um", "mae_flank_wear+adhesion_um"]
SHORT   = {"mae_overall_um": "overall", "mae_flank_wear_um": "flank",
           "mae_adhesion_um": "adh", "mae_flank_wear+adhesion_um": "f+a"}


def load_runs(run_dir: Path) -> list[dict]:
    rows = []
    for rj in sorted(run_dir.glob("*/results.json")):
        d = json.loads(rj.read_text())
        phys = d.get("physics", {})
        for split, m in d.get("splits", {}).items():
            rows.append({
                "family": run_dir.name,
                "experiment": d.get("experiment", rj.parent.name),
                "split": split,
                "lambda_max": phys.get("lambda_max", 0.0),
                "one_sided": phys.get("one_sided", False),
                "apply_to": phys.get("apply_to", "-"),
                "best_epoch": d.get("best_epoch"),
                **{SHORT[k]: m.get(k) for k in METRICS},
                "n_total": m.get("n_total"),
            })
    return rows


def show_family(df: pd.DataFrame, family: str, split: str = "test") -> None:
    sub = df[(df.family == family) & (df.split == split)].copy()
    if sub.empty:
        return
    ctrl = sub[sub.experiment.str.contains("ctrl")]
    base = ctrl.iloc[0] if not ctrl.empty else None
    sub = sub.sort_values("overall")
    print(f"\n{'='*92}\n  {family}  —  {split} set" + (f"   (Δ vs {base.experiment})" if base is not None else ""))
    print('='*92)
    hdr = f"  {'experiment':24s} {'λ':>5} {'1side':>5} {'ep':>3} " \
          f"{'overall':>8} {'flank':>7} {'adh':>7} {'f+a':>7}"
    if base is not None:
        hdr += f"   {'Δover':>6} {'Δadh':>6} {'Δf+a':>6}"
    print(hdr)
    for _, r in sub.iterrows():
        line = (f"  {r.experiment:24s} {r.lambda_max:>5.3g} "
                f"{str(bool(r.one_sided)):>5} {str(r.best_epoch):>3} "
                f"{r['overall']:>8.2f} {r['flank']:>7.2f} {r['adh']:>7.2f} {r['f+a']:>7.2f}")
        if base is not None:
            line += (f"   {r['overall']-base['overall']:>+6.2f} "
                     f"{r['adh']-base['adh']:>+6.2f} {r['f+a']-base['f+a']:>+6.2f}")
        print(line)
    print("  (negative Δ = physics improved over control)")


def main():
    dirs = [Path(p) for p in sys.argv[1:]]
    if not dirs:
        sys.exit("Usage: python aggregate_stage3.py <run_dir> [<run_dir> ...]")
    rows = []
    for d in dirs:
        if d.exists():
            rows += load_runs(d)
        else:
            print(f"[warn] {d} not found")
    if not rows:
        sys.exit("No results.json found.")
    df = pd.DataFrame(rows)
    for fam in df.family.unique():
        for split in ["test", "val"]:
            show_family(df, fam, split)
    out = dirs[0].parent / "stage3_summary.csv"
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
