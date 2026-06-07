import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MATWI_DATA_PATH = os.path.join(BASE_DIR, "..", "matwi")

WEAR_TYPES  = ["flank_wear", "adhesion", "flank_wear+adhesion"]

# Wear types containing adhesion — used for oversampling
ADHESION_TYPES = {"adhesion", "flank_wear+adhesion"}

WEAR_CAP    = 450.0

# Wear normalisation divisor (matches paper: wear in mm = wear_µm / 1000)
WEAR_DIVISOR   = 1000.0

# Number of windows used for within-pass segmented features
N_WINDOWS  = 10

# Number of lowest-SensorID passes used to define the per-set baseline
BASELINE_K = 3

MATERIAL_CK45 = "CK45"
MATERIAL_RVS  = "RVS 304"

# ── Official paper train/val/test split ───────────────────────────────────────
SPLIT_SETS = {
    "train":  [1, 2, 5, 7, 8, 10, 11],
    "val":    [3, 6, 12],
    "test":   [4, 9, 13],
    "unseen": [14, 15, 16, 17],
}

# ── Precomputed normalisation stats ──────────────────────────────────────────
# Computed over cropped images from the respective set ranges.
# Format: RGB mean and std in [0, 1] range.
NORM_STATS = {
    "1-13": {
        "mean": [0.40560478076171874, 0.39624402888997395, 0.5083500896809896],
        "std":  [0.19804957525771552, 0.1920124143538162,  0.2173112347091954],
    },
    "1-17": {
        "mean": [0.42896144165039063, 0.4173878169759115,  0.5292330250651042],
        "std":  [0.20020134730149317, 0.19364826735868443, 0.21420715969306942],
    },
    "imagenet": {
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
    },
}

# ── Crop coordinates per set (left, top, right, bottom) ──────────────────────
# From Table 2 of the MATWI paper. Applied to full 5496×3672 images.
# Produces 600×400 pixel crops as stated in the paper.
SET_CROPS = {
    1:  (2470, 1000, 3070, 1400),
    2:  (1150, 670,  1750, 1070),
    3:  (1150, 670,  1750, 1070),
    4:  (1150, 670,  1750, 1070),
    5:  (1150, 670,  1750, 1070),
    6:  (1150, 670,  1750, 1070),
    7:  (1150, 670,  1750, 1070),
    8:  (1150, 670,  1750, 1070),
    9:  (1150, 670,  1750, 1070),
    10: (1150, 670,  1750, 1070),
    11: (1150, 670,  1750, 1070),
    12: (1150, 670,  1750, 1070),
    13: (1150, 670,  1750, 1070),
    14: (1790, 2030, 2390, 2430),
    15: (1790, 2030, 2390, 2430),
    16: (1790, 2045, 2390, 2445),
    17: (1790, 2045, 2390, 2445),
}

# Sensor channel names — order matches CSV columns (no header in sensor files)
SENSOR_CHANNELS = ["acc", "acoustic", "Fx", "Fy", "Fz"]

# Sampling rate: 1 sample every 0.6 ms → 1666.67 Hz
SENSOR_FS = 1000.0 / 0.6  # ≈ 1666.67 Hz

# Total number of sensor features extracted per sample
N_SENSOR_FEATURES = len(SENSOR_CHANNELS) * 8  # 5 channels × 8 features = 40

# Type-label canonicalisation (matches DatasetClass_VisionSensors v2)
TYPE_MAP = {
    "adhesive":                 "adhesion",
    "adhesive_wear":            "adhesion",
    "flank":                    "flank_wear",
    "flank_wear_and_adhesion":  "flank_wear+adhesion",
    "flank_wear_&_adhesion":    "flank_wear+adhesion",
    "flank_wear+adhesive_wear": "flank_wear+adhesion",
    "combination":              "flank_wear+adhesion",
}

# ── Air-cut removal parameters (wavelet edge detector) ────────────────────────
# Ported from the previous group's RemoveAircuts/Removing_Aircuts.ipynb. Each
# recording is ONE cut (tool approach in air -> cutting -> retract in air); the
# detector finds a single [start, end) cutting window via a db4 level-4 Discrete
# Wavelet Transform of the vibration (acc) and acoustic channels and crops to it.
WAVELET_NAME       = "db4"
WAVELET_LEVEL      = 4
WAVELET_Z_THRESH   = 10.0     # detail-energy z-score for "active" (their FIXED_SENSITIVITY)
WAVELET_BASELINE_N = 1000     # first N detail coeffs assumed to be air (approach)
WAVELET_MIN_ACTIVE = 1000     # < this many active samples -> keep the full signal
WAVELET_TRIM       = -0.05    # negative => expand window outward 5% each side (their FIXED_TRIM)

# Cutting-parameter feature columns (from sets.csv).
# LGBM consumes NaN natively (split routes missing to a side); Ridge gets
# train-mean imputation in fit_predict_ridge.
CUTTING_FEATURE_NAMES = ["Vc", "n", "fz", "Vf", "Ae", "Ap", "z", "material_RVS"]

# ══════════════════════════════════════════════════════════════════════════════
# Feature set definitions
# ══════════════════════════════════════════════════════════════════════════════
#
# Extraction order per channel (8 features):
#   [mean, std, rms, p2p, kurtosis, dom_freq, low_energy, mid_energy]
#
# Channel layout (5 channels × 8 = 40 features):
#   acc:      indices  0– 7   (mean=0,  std=1,  rms=2,  p2p=3,  kurt=4,  dom=5,  low=6,  mid=7)
#   acoustic: indices  8–15   (mean=8,  std=9,  rms=10, p2p=11, kurt=12, dom=13, low=14, mid=15)
#   Fx:       indices 16–23   (mean=16, std=17, rms=18, p2p=19, kurt=20, dom=21, low=22, mid=23)
#   Fy:       indices 24–31   (mean=24, std=25, rms=26, p2p=27, kurt=28, dom=29, low=30, mid=31)
#   Fz:       indices 32–39   (mean=32, std=33, rms=34, p2p=35, kurt=36, dom=37, low=38, mid=39)
#

FEATURE_SETS = {
    # All 40 engineered features (time + frequency domain)
    "all40": list(range(40)),

    # 25 time-domain features only — no FFT/spectral engineering
    # First 5 features of each 8-feature block (mean, std, rms, p2p, kurtosis)
    "raw25": [
        0,  1,  2,  3,  4,     # acc:      mean, std, rms, p2p, kurtosis
        8,  9,  10, 11, 12,    # acoustic: mean, std, rms, p2p, kurtosis
        16, 17, 18, 19, 20,    # Fx:       mean, std, rms, p2p, kurtosis
        24, 25, 26, 27, 28,    # Fy:       mean, std, rms, p2p, kurtosis
        32, 33, 34, 35, 36,    # Fz:       mean, std, rms, p2p, kurtosis
    ],

    # Top 25 features by Ridge regression |coefficient| from sensor sanity check.
    # Drops all 5 dom_freq features (consistently weakest), all 5 kurtosis
    # features, and 5 additional weak features: acc_mean, acc_low_energy,
    # Fz_mean, Fz_p2p, acoustic_mean.
    # This is a mix of time-domain and frequency-domain features.
    "top25": [
        1,  2,  3,  7,             # acc:      std, rms, p2p, mid_energy
        9,  10, 11, 14, 15,        # acoustic: std, rms, p2p, low_energy, mid_energy
        16, 17, 18, 19, 22, 23,    # Fx:       mean, std, rms, p2p, low_energy, mid_energy
        24, 25, 26, 27, 30, 31,    # Fy:       mean, std, rms, p2p, low_energy, mid_energy
        33, 34, 38, 39,            # Fz:       std, rms, low_energy, mid_energy
    ],
}