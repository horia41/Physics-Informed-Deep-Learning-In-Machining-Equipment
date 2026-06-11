from typing import Any

def fmt_metrics_v1(m: dict[str, Any]) -> str:
    return (
        f"overall={m['mae_overall_um']:.2f}µm  "
        f"flank={m['mae_flank_wear_um']:.2f}  "
        f"adh={m['mae_adhesion_um']:.2f}  "
        f"f+a={m['mae_flank_wear+adhesion_um']:.2f}  "
        f"(n={m['n_total']})"
    )

def fmt_metrics_v2(label: str, m: dict) -> str:
    return (f"  [{label:24s}]  overall={m['overall']:5.1f}µm  "
            f"flank={m.get('flank_wear', float('nan')):5.1f}  "
            f"adh={m.get('adhesion', float('nan')):5.1f}  "
            f"f+a={m.get('flank_wear+adhesion', float('nan')):5.1f}  "
            f"(n={m['n']})")