from pathlib import Path

# PATHS
ROOT       = Path(__file__).parent.parent   # master_thesis/
DATA_DIR   = ROOT / "Data"
LABELS_CSV = DATA_DIR / "labels.csv"
SETS_CSV   = DATA_DIR / "sets.csv"
SETS_DIR   = DATA_DIR / "sets"

MODELS_DIR  = ROOT / "code" / "outputs" / "models"
FIGURES_DIR = ROOT / "code" / "outputs" / "figures"

# DATASET SPLITS (from paper)
TRAIN_SETS = [1, 2, 5, 7, 8, 10, 11]
VAL_SETS   = [3, 6, 12]
TEST_SETS  = [4, 9, 13]

# IMAGE SETTINGS
IMAGE_SIZE = (224, 224)   # ResNet expects 224x224
IMAGE_CHANNELS = 3        # RGB

# TRAINING HYPERPARAMETERS
BATCH_SIZE    = 32
LEARNING_RATE = 3e-4      # value used in the paper
N_EPOCHS      = 50
PATIENCE      = 10        # early stopping

# WEAR NORMALIZATION
WEAR_MAX = 300.0          # approximate max wear in µm, used to normalize labels to [0,1]
