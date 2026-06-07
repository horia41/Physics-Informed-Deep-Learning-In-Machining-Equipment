
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from utils.console_utils import (
    section,
    subsection
)

from dataset.matwi.dataset_loader import (
    get_split_label,
    load_one_sensor_file
)

warnings.filterwarnings("ignore")


_PALETTE = {
    "flank_wear":             "#4C72B0",
    "adhesion":               "#DD8452",
    "flank_wear+adhesion":    "#55A868",
    "unknown":                "#C44E52",
    "CK45":                   "#4C72B0",
    "RVS 304":                "#DD8452",
}
_WEAR_TYPE_ORDER = ["flank_wear", "adhesion", "flank_wear+adhesion"]


_SPLIT_COLOR = {"train": "#4C72B0", "val": "#55A868",
               "test": "#DD8452", "unseen": "#8172B2"}

def _save(fig: plt.Figure, out_dir: Path, name: str, dpi: int = 150) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path.name}")



def analyse_integrity(df: pd.DataFrame, data_dir: Path, out_dir: Path) -> None:
    section("1. DATASET INTEGRITY")

    total        = len(df)
    n_image_only = (df["has_image"] & ~df["has_sensor"]).sum()
    n_sensor_only= (~df["has_image"] & df["has_sensor"]).sum()
    n_both       = df["both"].sum()
    n_neither    = (~df["has_image"] & ~df["has_sensor"]).sum()

    print(f"  Total rows in labels.csv : {total}")
    print(f"  Both image + sensor      : {n_both}  ({100*n_both/total:.1f}%)")
    print(f"  Image only (no sensor)   : {n_image_only}")
    print(f"  Sensor only (no image)   : {n_sensor_only}")
    print(f"  Neither                  : {n_neither}")
    print(f"  Missing wear values      : {df['wear'].isna().sum()}")
    print(f"  Wear range               : {df['wear'].min():.1f} – {df['wear'].max():.1f} µm")

    subsection("Sync errors per set")
    sync = (df.groupby("Set")
              .agg(total=("Set","count"),
                   both=("both","sum"),
                   img_only=("has_image", lambda x: (x & ~df.loc[x.index,"has_sensor"]).sum()),
                   sen_only=("has_sensor", lambda x: (~df.loc[x.index,"has_image"] & x).sum()))
              .reset_index())
    sync["sync_error"] = sync["total"] - sync["both"]
    print(sync.to_string(index=False))

    # -- Figure: stacked bar of data availability per set --
    fig, ax = plt.subplots(figsize=(14, 4))
    sets = sync["Set"].values
    ax.bar(sets, sync["both"],     label="Both",        color="#4C72B0")
    ax.bar(sets, sync["img_only"], bottom=sync["both"], label="Image only", color="#DD8452")
    ax.bar(sets, sync["sen_only"],
           bottom=sync["both"]+sync["img_only"],        label="Sensor only",color="#55A868")
    ax.set_xlabel("Set"); ax.set_ylabel("Row count")
    ax.set_title("Data availability per set (sync errors highlighted)")
    ax.set_xticks(sets)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "01_data_availability")

    # -- Check physical files exist --
    subsection("Physical file existence check")
    if "ImageFile" in df.columns:
        img_df = df[df["has_image"]].copy()
        img_df["img_exists"] = img_df["ImageFile"].apply(
            lambda p: (data_dir / str(p)).exists() if pd.notna(p) else False
        )
        n_found = img_df["img_exists"].sum()
        print(f"  Images referenced: {len(img_df)} | Found on disk: {n_found} "
              f"| Missing: {len(img_df)-n_found}")

    if "SensorFile" in df.columns:
        sen_df = df[df["has_sensor"]].copy()
        sen_df["sen_exists"] = sen_df["SensorFile"].apply(
            lambda p: (data_dir / str(p)).exists() if pd.notna(p) else False
        )
        n_found = sen_df["sen_exists"].sum()
        print(f"  Sensor files referenced: {len(sen_df)} | Found on disk: {n_found} "
              f"| Missing: {len(sen_df)-n_found}")




def analyse_labels(df: pd.DataFrame, out_dir: Path) -> None:
    section("2. LABEL ANALYSIS")

    valid = df[df["wear"].notna()].copy()

    subsection("Wear statistics (µm)")
    stats = valid["wear"].describe(percentiles=[.1,.25,.5,.75,.9])
    print(stats.to_string())

    subsection("Wear type counts")
    type_counts = valid["type"].value_counts()
    print(type_counts.to_string())

    subsection("Wear type × material")
    print(pd.crosstab(valid["type"], valid["material"]).to_string())

    # ── Figure 1: wear distribution overview (4 panels) ──────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Panel A — overall histogram
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(valid["wear"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax0.axvline(valid["wear"].median(), color="red", linestyle="--", label=f"Median {valid['wear'].median():.0f} µm")
    ax0.set_xlabel("Wear (µm)"); ax0.set_ylabel("Count")
    ax0.set_title("A — Overall wear distribution")
    ax0.legend(); ax0.grid(axis="y", alpha=0.3)

    # Panel B — by wear type
    ax1 = fig.add_subplot(gs[0, 1])
    for wt in _WEAR_TYPE_ORDER:
        sub = valid[valid["type"] == wt]["wear"]
        if len(sub):
            ax1.hist(sub, bins=30, alpha=0.6,
                     color=_PALETTE.get(wt, "grey"),
                     label=f"{wt} (n={len(sub)})", edgecolor="white")
    ax1.set_xlabel("Wear (µm)"); ax1.set_ylabel("Count")
    ax1.set_title("B — Wear by wear type")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    # Panel C — by material
    ax2 = fig.add_subplot(gs[1, 0])
    for mat in ["CK45", "RVS 304"]:
        sub = valid[valid["material"] == mat]["wear"]
        ax2.hist(sub, bins=30, alpha=0.65,
                 color=_PALETTE[mat], label=f"{mat} (n={len(sub)})", edgecolor="white")
    ax2.set_xlabel("Wear (µm)"); ax2.set_ylabel("Count")
    ax2.set_title("C — Wear by material")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    # Panel D — boxplot by wear type
    ax3 = fig.add_subplot(gs[1, 1])
    data_by_type = [valid[valid["type"] == wt]["wear"].dropna().values
                    for wt in _WEAR_TYPE_ORDER]
    bp = ax3.boxplot(data_by_type, patch_artist=True, notch=False,
                     medianprops=dict(color="black", linewidth=2))
    for patch, wt in zip(bp["boxes"], _WEAR_TYPE_ORDER):
        patch.set_facecolor(_PALETTE.get(wt, "grey"))
        patch.set_alpha(0.7)
    ax3.set_xticks(range(1, len(_WEAR_TYPE_ORDER)+1))
    ax3.set_xticklabels([wt.replace("_", "\n") for wt in _WEAR_TYPE_ORDER], fontsize=8)
    ax3.set_ylabel("Wear (µm)")
    ax3.set_title("D — Wear boxplot by type")
    ax3.grid(axis="y", alpha=0.3)

    fig.suptitle("MATWI — Wear Label Analysis", fontsize=14, fontweight="bold")
    _save(fig, out_dir, "02_wear_distribution")

    # ── Figure 2: wear type per set (stacked bar) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    pivot = (valid.groupby(["Set", "type"])
                  .size()
                  .unstack(fill_value=0)
                  .reindex(columns=_WEAR_TYPE_ORDER, fill_value=0))
    bottom = np.zeros(len(pivot))
    for wt in _WEAR_TYPE_ORDER:
        if wt in pivot.columns:
            ax.bar(pivot.index, pivot[wt], bottom=bottom,
                   color=_PALETTE.get(wt, "grey"), label=wt, alpha=0.85)
            bottom += pivot[wt].values

    # Shade by split
    for set_num in pivot.index:
        split = get_split_label(set_num)
        ax.axvspan(set_num - 0.4, set_num + 0.4,
                   color=_SPLIT_COLOR[split], alpha=0.08, zorder=0)

    # Material separator
    ax.axvline(11.5, color="black", linestyle=":", linewidth=1.5, label="CK45 | RVS 304")

    # Legend
    type_patches = [mpatches.Patch(color=_PALETTE[wt], label=wt) for wt in _WEAR_TYPE_ORDER]
    split_patches = [mpatches.Patch(color=_SPLIT_COLOR[s], alpha=0.4, label=f"split:{s}")
                     for s in _SPLIT_COLOR]
    ax.legend(handles=type_patches + split_patches, fontsize=8,
              loc="upper left", ncol=2)
    ax.set_xlabel("Set"); ax.set_ylabel("Count")
    ax.set_xticks(pivot.index)
    ax.set_title("Wear type distribution per set  (background = paper train/val/test split)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "03_wear_type_per_set")



def analyse_trajectories(df: pd.DataFrame, out_dir: Path) -> None:
    section("3. WEAR TRAJECTORIES PER SET")

    valid = df[df["wear"].notna() & df["has_image"]].copy()
    valid = valid.sort_values(["Set", "ImageID"])

    subsection("Per-set wear stats")
    traj_stats = (valid.groupby("Set")["wear"]
                       .agg(["count", "min", "max", "mean", "std"])
                       .round(1))
    traj_stats["material"] = traj_stats.index.map(
        lambda s: "CK45" if s <= 11 else "RVS 304"
    )
    traj_stats["split"] = traj_stats.index.map(get_split_label)
    print(traj_stats.to_string())

    # ── Figure: 17 subplots, one per set ─────────────────────────────────────
    fig, axes = plt.subplots(4, 5, figsize=(20, 14), sharey=False)
    axes = axes.flatten()

    for idx, set_num in enumerate(sorted(valid["Set"].unique())):
        ax = axes[idx]
        sub = valid[valid["Set"] == set_num].sort_values("ImageID")
        mat = "CK45" if set_num <= 11 else "RVS 304"
        split = get_split_label(set_num)

        # Color points by wear type
        for wt in _WEAR_TYPE_ORDER:
            wt_sub = sub[sub["type"] == wt]
            if len(wt_sub):
                ax.scatter(wt_sub["ImageID"], wt_sub["wear"],
                           c=_PALETTE.get(wt, "grey"), s=18, alpha=0.8,
                           zorder=3, label=wt)

        # Line connecting measurements
        ax.plot(sub["ImageID"], sub["wear"],
                color="grey", linewidth=0.8, alpha=0.5, zorder=2)

        ax.set_title(f"Set {set_num}  [{mat}]\n{split}  n={len(sub)}",
                     fontsize=8)
        ax.set_xlabel("Image ID", fontsize=7)
        ax.set_ylabel("Wear (µm)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    # Hide unused subplots (we have 17 sets, 4×5=20 panels)
    for idx in range(len(valid["Set"].unique()), len(axes)):
        axes[idx].set_visible(False)

    # Shared legend
    legend_elements = [Line2D([0],[0], marker="o", color="w",
                               markerfacecolor=_PALETTE[wt], label=wt, markersize=7)
                       for wt in _WEAR_TYPE_ORDER]
    fig.legend(handles=legend_elements, loc="lower right",
               fontsize=9, ncol=1, title="Wear type")

    fig.suptitle("Wear progression per set (ImageID = time proxy)",
                 fontsize=13, fontweight="bold")
    _save(fig, out_dir, "04_wear_trajectories", dpi=150)

    # ── Figure: CK45 vs RVS304 mean trajectories ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, mat in zip(axes, ["CK45", "RVS 304"]):
        mat_df = valid[valid["material"] == mat]
        for set_num in sorted(mat_df["Set"].unique()):
            sub = mat_df[mat_df["Set"] == set_num].sort_values("ImageID")
            # Normalise x to [0,1] so all tools align regardless of length
            x_norm = (sub["ImageID"] - sub["ImageID"].min()) / \
                     max(sub["ImageID"].max() - sub["ImageID"].min(), 1)
            ax.plot(x_norm, sub["wear"],
                    alpha=0.5, linewidth=1.2, label=f"Set {set_num}")
        ax.set_title(f"Material: {mat}")
        ax.set_xlabel("Normalised tool life (0=new, 1=failed)")
        ax.set_ylabel("Wear (µm)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    fig.suptitle("Wear curves: CK45 vs RVS 304 (x-axis normalised to tool life)",
                 fontsize=12, fontweight="bold")
    _save(fig, out_dir, "05_material_comparison_trajectories")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Cutting parameters analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyse_cutting_params(df: pd.DataFrame, sets_df: pd.DataFrame,
                           out_dir: Path) -> None:
    if sets_df.empty:
        print("\nSKIP No sets.csv found — cutting parameter analysis skipped.")
        return

    section("4. CUTTING PARAMETERS")

    # Merge max wear per set with cutting parameters
    wear_per_set = (df[df["wear"].notna()]
                    .groupby("Set")["wear"]
                    .agg(["max", "mean", "count"])
                    .reset_index())
    wear_per_set.columns = ["Set", "max_wear", "mean_wear", "n_samples"]

    sets_df_reset = sets_df.reset_index()
    sets_df_reset["Set_num"] = sets_df_reset.iloc[:, 0].str.extract(r"(\d+)").astype(int)
    merged = wear_per_set.merge(sets_df_reset, left_on="Set", right_on="Set_num", how="left")

    subsection("Sets with unknown cutting parameters (Set 1)")
    unknown = merged[merged["Vc"].isna()]
    print(f"  {len(unknown)} set(s) with unknown params: {unknown['Set'].tolist()}")

    subsection("Sets.csv — overview")
    print(sets_df.to_string())

    subsection("Notable edge cases")
    print("  Set 1  : Vc/n/fz/Vf all unknown — cannot use Taylor's eq for this set")
    print("  Set 6  : Vf listed as '185/148' — parameter change mid-run")
    print("  Set 17 : z=2 (two inserts) vs z=1 for all others — different Taylor constant C")

    # ── Figure: cutting speed vs max wear ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    params = [("Vc", "Cutting speed Vc (m/min)"),
              ("fz", "Feed per tooth fz (mm)"),
              ("Vf", "Feed rate Vf (mm/min)")]

    for ax, (param, label) in zip(axes, params):
        sub = merged[merged[param].notna()]
        for mat in ["CK45", "RVS 304"]:
            ms = sub[sub["material"] == mat]
            ax.scatter(ms[param], ms["max_wear"],
                       color=_PALETTE[mat], s=80, alpha=0.8,
                       edgecolors="black", linewidth=0.5, label=mat)
            # Annotate set numbers
            for _, row in ms.iterrows():
                ax.annotate(f"S{int(row['Set'])}", (row[param], row["max_wear"]),
                            fontsize=7, ha="center", va="bottom")
        ax.set_xlabel(label); ax.set_ylabel("Max wear (µm)")
        ax.set_title(f"{param} vs max wear")
        ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("Cutting parameters vs maximum wear reached per set",
                 fontsize=12, fontweight="bold")
    _save(fig, out_dir, "06_cutting_params_vs_wear")

    # ── Taylor's equation feasibility note ────────────────────────────────────
    subsection("Taylor's equation feasibility")
    print("  Taylor: V · T^n = C")
    print("  Variables available per set:")
    print("    V (Vc) — available for all sets except Set 1")
    print("    T      — approximated by ImageID at failure (last measurement)")
    print("    n, C   — to be fitted from data (material-dependent constants)")
    print("  Set 17 (z=2) uses two inserts → wear dynamics differ → fit separately")
    print("  Set 6 has varying Vf mid-run → use with caution in Taylor fitting")



def analyse_sensors(df: pd.DataFrame, data_dir: Path, out_dir: Path) -> None:
    section("5. SENSOR ANALYSIS")

    if "SensorFile" not in df.columns:
        print("  SKIP SensorFile column not in labels.csv — sensor analysis skipped.")
        return

    sensor_rows = df[df["has_sensor"] & df["SensorFile"].notna()].copy()

    subsection("Sampling mismatch: sensor samples per image")
    # Count how many sensor files exist per set as a proxy
    # (each sensor file = one milling pass = one image measurement)
    sen_per_set = sensor_rows.groupby("Set")["SensorName"].count()
    print("  Sensor readings per set:")
    print(sen_per_set.to_string())
    print(f"\n  Total sensor files: {len(sensor_rows)}")
    print(f"  Each sensor file ≈ 1 milling pass, sampled at 0.6 ms intervals")
    print(f"  → ~1,666 samples/second × pass duration ≈ tens of thousands of rows per file")

    # Try to load a few sensor files for signal analysis
    subsection("Loading sample sensor files for signal inspection")

    # Pick one file from an early, middle, and late measurement from Set 5
    # (Set 5 has notably high wear — good for seeing signal evolution)
    set5_sensors = sensor_rows[sensor_rows["Set"] == 5].sort_values("SensorID")

    sample_indices = []
    if len(set5_sensors) >= 3:
        sample_indices = [
            set5_sensors.iloc[0],
            set5_sensors.iloc[len(set5_sensors)//2],
            set5_sensors.iloc[-1],
        ]

    loaded_samples = []
    for row in sample_indices:
        fpath = data_dir / str(row["SensorFile"])
        if fpath.exists():
            sdf = load_one_sensor_file(fpath)
            if sdf is not None:
                sdf["wear_at_capture"] = row["wear"]
                img_id = int(row['ImageID']) if pd.notna(row['ImageID']) else "?"
                wear_val = f"{row['wear']:.0f}" if pd.notna(row['wear']) else "?"
                sdf["label"] = f"ImageID={img_id} | wear={wear_val}µm"
                loaded_samples.append(sdf)
                print(f"  Loaded: {fpath.name}  → {len(sdf)} rows")
        else:
            print(f"  MISSING {fpath}")

    if not loaded_samples:
        print("  SKIP No sensor files found on disk — signal plots skipped.")
        print("         Run the download script first, then re-run EDA.")
        _plot_sensor_placeholder(out_dir)
        return

    # ── Figure: raw sensor signals for 3 wear stages ──────────────────────────
    channels = ["acc", "acoustic", "Fx", "Fy", "Fz"]
    ch_labels = ["Accelerometer", "Acoustic Emission", "Force X", "Force Y", "Force Z"]
    colors    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, axes = plt.subplots(len(channels), len(loaded_samples),
                             figsize=(16, 12), sharex="col")
    if len(loaded_samples) == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, sdf in enumerate(loaded_samples):
        t = np.arange(len(sdf)) * 0.0006   # 0.6 ms per sample → seconds
        for row_idx, (ch, ch_label, color) in enumerate(zip(channels, ch_labels, colors)):
            ax = axes[row_idx, col_idx]
            ax.plot(t, sdf[ch], color=color, linewidth=0.4, alpha=0.8)
            if row_idx == 0:
                ax.set_title(sdf["label"].iloc[0], fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(ch_label, fontsize=8)
            if row_idx == len(channels) - 1:
                ax.set_xlabel("Time (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)

    fig.suptitle("Raw sensor signals — early / mid / late in tool life (Set 5)",
                 fontsize=12, fontweight="bold")
    _save(fig, out_dir, "07_sensor_signals_raw", dpi=120)

    # ── Figure: sensor statistics across files ────────────────────────────────
    subsection("Computing per-file sensor statistics (RMS, mean, std)")

    # Sample up to 30 sensor files spread across all sets for efficiency
    sample_rows = []
    for set_num, group in sensor_rows.groupby("Set"):
        sample_rows.append(group.sample(min(len(group), 3), random_state=42))
    sample_rows = pd.concat(sample_rows, ignore_index=True)

    stats_records = []
    for _, row in sample_rows.iterrows():
        fpath = data_dir / str(row["SensorFile"])
        if not fpath.exists():
            continue
        sdf = load_one_sensor_file(fpath)
        if sdf is None:
            continue
        rec = {"Set": row["Set"], "wear": row["wear"],
               "material": row["material"], "n_samples": len(sdf)}
        for ch in channels:
            if ch in sdf.columns:
                vals = sdf[ch].dropna().values
                rec[f"{ch}_rms"]  = float(np.sqrt(np.mean(vals**2)))
                rec[f"{ch}_mean"] = float(np.mean(vals))
                rec[f"{ch}_std"]  = float(np.std(vals))
        stats_records.append(rec)

    if not stats_records:
        print("  SKIP No sensor files found for statistics.")
        return

    stats_df = pd.DataFrame(stats_records)
    print(f"  Computed stats for {len(stats_df)} sensor files")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    rms_cols = [f"{ch}_rms" for ch in channels if f"{ch}_rms" in stats_df.columns]
    axes_flat = axes.flatten()

    for idx, (ch, ax) in enumerate(zip(channels, axes_flat)):
        rms_col = f"{ch}_rms"
        if rms_col not in stats_df.columns:
            continue
        for mat in ["CK45", "RVS 304"]:
            ms = stats_df[stats_df["material"] == mat]
            ax.scatter(ms["wear"], ms[rms_col],
                       color=_PALETTE[mat], alpha=0.7, s=40,
                       edgecolors="black", linewidth=0.3, label=mat)
        ax.set_xlabel("Wear (µm)"); ax.set_ylabel("RMS")
        ax.set_title(f"{ch_labels[idx]} RMS vs wear")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Hide unused panel
    for idx in range(len(channels), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Sensor RMS vs wear level (by material)",
                 fontsize=12, fontweight="bold")
    _save(fig, out_dir, "08_sensor_rms_vs_wear")

    subsection("Force Z channel check")
    fz_means = [r.get("Fz_mean", None) for r in stats_records]
    fz_means = [v for v in fz_means if v is not None]
    if fz_means:
        print(f"  Force Z mean across {len(fz_means)} files: "
              f"mean={np.mean(fz_means):.4f}, std={np.std(fz_means):.4f}")
        print("  → Force Z should be ~0 throughout (spindle moves only in X/Y)")
        print(f"  → {'Confirmed near-zero' if abs(np.mean(fz_means)) < 1 else 'Non-zero — investigate'}")


def _plot_sensor_placeholder(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5,
            "Sensor files not found on disk.\n"
            "Run download_data.py first, then re-run EDA.py.",
            ha="center", va="center", fontsize=14, color="grey",
            transform=ax.transAxes)
    ax.set_axis_off()
    _save(fig, out_dir, "07_sensor_signals_placeholder")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Image analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyse_images(df: pd.DataFrame, data_dir: Path, out_dir: Path) -> None:
    section("6. IMAGE ANALYSIS")

    if "ImageFile" not in df.columns:
        print("  SKIP ImageFile column not found.")
        return

    img_rows = df[df["has_image"] & df["ImageFile"].notna()].copy()

    try:
        from PIL import Image as PILImage
    except ImportError:
        print("  SKIP Pillow not installed — image display skipped.")
        print("         pip install Pillow")
        return

    wear_low    = img_rows[img_rows["wear"] <  80].dropna(subset=["wear"])
    wear_mid    = img_rows[(img_rows["wear"] >= 80)  & (img_rows["wear"] < 200)].dropna(subset=["wear"])
    wear_high   = img_rows[img_rows["wear"] >= 200].dropna(subset=["wear"])
    wear_adhes  = img_rows[img_rows["type"].isin(["adhesion","flank_wear+adhesion"])]

    stages = [
        ("Low wear (<80 µm)",        wear_low,   "#4C72B0"),
        ("Mid wear (80–200 µm)",     wear_mid,   "#DD8452"),
        ("High wear (≥200 µm)",      wear_high,  "#C44E52"),
        ("Adhesion",                 wear_adhes, "#55A868"),
    ]

    fig, axes = plt.subplots(len(stages), 4, figsize=(18, 14))

    images_found = 0
    for row_idx, (stage_label, stage_df, _) in enumerate(stages):
        # Pick up to 4 representative samples
        samples = stage_df.sample(min(4, len(stage_df)), random_state=42) if len(stage_df) else pd.DataFrame()
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            if col_idx < len(samples):
                sample = samples.iloc[col_idx]
                fpath  = data_dir / str(sample["ImageFile"])
                if fpath.exists():
                    try:
                        img = PILImage.open(fpath)
                        ax.imshow(np.array(img), aspect="auto")
                        ax.set_title(
                            f"Set {sample['Set']} | wear={sample['wear']:.0f}µm\n"
                            f"{sample['type']}",
                            fontsize=7
                        )
                        images_found += 1
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Load error:\n{e}",
                                ha="center", va="center", fontsize=7, transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "File not\nfound",
                            ha="center", va="center", fontsize=9,
                            color="grey", transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        fontsize=14, color="lightgrey", transform=ax.transAxes)
            ax.axis("off")

        # Row label on the left
        axes[row_idx, 0].set_ylabel(stage_label, fontsize=10, rotation=90,
                                     labelpad=10)

    if images_found == 0:
        print("  SKIP No image files found on disk — image grid skipped.")
        print("         Run download_data.py first.")
        plt.close(fig)
        _plot_image_placeholder(out_dir)
        return

    fig.suptitle("Sample images at different wear stages",
                 fontsize=13, fontweight="bold")
    _save(fig, out_dir, "09_image_samples", dpi=120)

    # ── Image size consistency check ──────────────────────────────────────────
    subsection("Image dimension check (sample of 20 images)")
    sample20 = img_rows.sample(min(20, len(img_rows)), random_state=0)
    sizes = []
    for _, row in sample20.iterrows():
        fpath = data_dir / str(row["ImageFile"])
        if fpath.exists():
            try:
                img = PILImage.open(fpath)
                sizes.append(img.size)
            except Exception:
                pass
    if sizes:
        unique_sizes = set(sizes)
        print(f"  Sampled {len(sizes)} images → unique sizes: {unique_sizes}")
        if len(unique_sizes) == 1:
            print(f"  All consistent: {unique_sizes.pop()}")
        else:
            print(f"  Mixed sizes detected — check crop consistency across sets")


def _plot_image_placeholder(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5,
            "Image files not found on disk.\n"
            "Run download_data.py first, then re-run EDA.py.",
            ha="center", va="center", fontsize=14, color="grey",
            transform=ax.transAxes)
    ax.set_axis_off()
    _save(fig, out_dir, "09_image_samples_placeholder")



def analyse_modelling(df: pd.DataFrame, out_dir: Path) -> None:
    section("7. MODELLING IMPLICATIONS")

    valid = df[df["wear"].notna()].copy()

    # ── Train/val/test split review ───────────────────────────────────────────
    subsection("Paper's train/val/test split (sets 1–13 only)")
    print("  Train : Sets 1, 2, 5, 7, 8, 10, 11")
    print("  Val   : Sets 3, 6, 12")
    print("  Test  : Sets 4, 9, 13")
    print("  Unseen: Sets 14–17 (never used in paper baseline)")
    print()

    split_stats = (valid.groupby("split")
                        .agg(n_samples=("wear","count"),
                             n_sets=("Set","nunique"),
                             wear_mean=("wear","mean"),
                             wear_std=("wear","std"),
                             adhesion_frac=("type", lambda x: (x.isin(["adhesion","flank_wear+adhesion"])).mean()))
                        .round(2))
    print(split_stats.to_string())

    # ── Figure: split overview ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A — sample count and wear distribution per split
    ax = axes[0]
    for split_label, color in _SPLIT_COLOR.items():
        sub = valid[valid["split"] == split_label]["wear"]
        if len(sub):
            ax.hist(sub, bins=25, alpha=0.55, color=color,
                    label=f"{split_label} (n={len(sub)})", edgecolor="white")
    ax.set_xlabel("Wear (µm)"); ax.set_ylabel("Count")
    ax.set_title("Wear distribution by split")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # Panel B — adhesion fraction per split
    ax = axes[1]
    split_order = ["train", "val", "test", "unseen"]
    adhes_fracs = [
        valid[valid["split"] == s]["type"].isin(["adhesion","flank_wear+adhesion"]).mean()
        for s in split_order
    ]
    bars = ax.bar(split_order, [f*100 for f in adhes_fracs],
                  color=[_SPLIT_COLOR[s] for s in split_order], alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Adhesion samples (%)")
    ax.set_title("Adhesion wear fraction per split\n(adhesion = hard case for vision-only models)")
    ax.grid(axis="y", alpha=0.3)
    for bar, frac in zip(bars, adhes_fracs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{frac*100:.1f}%", ha="center", fontsize=10)

    fig.suptitle("Modelling split analysis", fontsize=12, fontweight="bold")
    _save(fig, out_dir, "10_split_analysis")

    # ── Imbalance / wear binning ───────────────────────────────────────────────
    subsection("Wear value distribution (imbalance check)")
    bins = [0, 50, 100, 150, 200, 300, 450, 1000]
    labels_b = ["0–50", "50–100", "100–150", "150–200", "200–300", "300–450", ">450"]
    valid["wear_bin"] = pd.cut(valid["wear"], bins=bins, labels=labels_b, right=False)
    bin_counts = valid["wear_bin"].value_counts().sort_index()
    print(f"\n  Wear bin distribution (µm):")
    for b, c in bin_counts.items():
        bar_str = "█" * int(c / bin_counts.max() * 30)
        print(f"  {str(b):>10}  {bar_str}  ({c})")