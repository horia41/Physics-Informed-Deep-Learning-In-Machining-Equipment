# Physics Informed Deep Learning In Machining Equipment (PINNs)

## 1. Overview of steps
Our project follows 4 stages:
1. **Vision-only baseline** — establish reference performance with images alone - **DONE**
2. **Multimodal sensor fusion** — test whether adding sensor data improves predictions - **DONE**
3. **Physics-informed loss (Taylor's equation)** — add physical constraints to the model - **NEXT**
4. **Refinement & compression** — hard constraints, pruning, quantisation for edge deployment - **NEXT**

---

## 2. Dataset 
- folder `dataset` handles everything in terms of data ; has `download_data.py` which automatically downloads the MATWI dataset and creates a folder `dataset/matwi` where it will be downloaded; has `EDA.py` which will output in a folder `dataset/eda_output` and performs exploratory data analysis on our dataset which will help us see details about our data and establish what data preprocessing steps and feature engineering we should perform; here, still missing the DataLoader part
- created folders `vision-baseline`, `vision-sensor` and `pinn` where we'll develop each model accordingly so we dont necessarily have to create multiple branches while using the same `dataset` folder

So, for now, make sure to run the `dataset/download_data.py` and then the `dataset/EDA.py` so you can see the plots and outputs of it.

**Source:** De Pauw et al. (2023), KU Leuven. 17 tools run to failure, ~100 measurements each.
 
**Key numbers:**
- 1,663 labelled samples total
- Each sample: one high-res image (5496×3672 px) + one sensor CSV (~99,000 rows × 5 channels)
- Wear range: 15–750 µm (capped at 450 µm for training, matching the paper)
- Wear normalised to [0, 1] by dividing by 1000
 
**Materials:**
- Sets 1–11: CK45 steel (mostly flank wear)
- Sets 12–17: RVS 304 stainless steel (heavy adhesive wear)
 
**Official split (sets 1–13):**
 
| Split   | Sets                    | Samples | Adhesion fraction | Notes                          |
|---------|-------------------------|---------|-------------------|--------------------------------|
| Train   | 1, 2, 5, 7, 8, 10, 11  | 664     | 14%               | CK45 only, almost no F+A      |
| Val     | 3, 6, 12                | 301     | 44%               | Includes RVS 304, hard split   |
| Test    | 4, 9, 13                | 254     | 24%               | Includes RVS 304               |
| Unseen  | 14, 15, 16, 17          | 444     | 52%               | RVS 304 only, never in paper   |
 
**The core challenge:** The training set has almost no flank_wear+adhesion (F+A) samples, 
but val/test are loaded with them. The paper's baseline gets 14 µm MAE on clean flank wear 
but 91 µm on F+A — that's the gap we're trying to close.
 
**Sync errors:** Not every sample has both image and sensor data. When we require both 
modalities (for sensor fusion), we lose 17 training samples (664 → 647), 1 val sample 
(301 → 300), and 7 test samples (254 → 247).
 
**Normalisation stats reference**
 
| set_range    | mean (RGB)               | std (RGB)                |
|--------------|--------------------------|--------------------------|
| `"1-13"`     | [0.4056, 0.3962, 0.5084] | [0.1980, 0.1920, 0.2173] |
| `"1-17"`     | [0.4290, 0.4174, 0.5292] | [0.2002, 0.1936, 0.2142] |
| `"imagenet"` | [0.485, 0.456, 0.406]    | [0.229, 0.224, 0.225]    |

 
**Train/val/test split reference**
 
| Split  | Sets                  | Material      | Notes                                                 |
|--------|-----------------------|---------------|-------------------------------------------------------|
| train  | 1, 2, 5, 7, 8, 10, 11 | CK45          | No `flank_wear+adhesion` samples                      |
| val    | 3, 6, 12              | CK45 + RVS304 | Hard split — 41% adhesion                             |
| test   | 4, 9, 13              | CK45 + RVS304 | 24% adhesion                                          |
| unseen | 14, 15, 16, 17        | RVS304 only   | 52% adhesion — added to train when `set_range="1-17"` |

---

## 3. Code Structure
 
All code lives under the project root with this structure:
 
```
project_pinn(or however you named it)/
├── dataset/
│   ├── download_data.py           # Downloads MATWI from KU Leuven Dataverse
│   ├── EDA.py                     # Exploratory data analysis (10 plots)
│   └── matwi/                     # Downloaded data (labels.csv, sets.csv, Set1–17/)
│
├── vision-only/                   # Stage 1 — Vision 
│   ├── DatasetClass_Vision.py     # Vision-only PyTorch Dataset
│   ├── ResNet_EfficientNet_Vision.py  # Model class (ResNet50 / EfficientNetV2)
│   ├── train_vision.py            # Vision ablation training script (9 experiments)
│   ├── gather_results.py          # Results aggregator
│   └── run_vision_ablation.sh     # SLURM job script for Snellius
│
├── vision-sensor/                 # Stage 2 — Vision & Sensor 
│   ├── DatasetClass_VisionSensors.py  # Multimodal Dataset (images + 40 sensor features)
│   ├── modelVisionSensor.py       # Multimodal model (early/intermediate/late fusion)
│   ├── train_vision_sensor.py     # Sensor fusion training script (10 experiments)
│   ├── sensor_sanity_check.py     # Ridge regression on sensor features alone
│   ├── gather_results_sensor.py   # Results aggregator
│   └── run_sensor_ablation.sh     # SLURM job script for Snellius
│
├── pinn/                          # Stage 3 — Taylor physics  (TO DO)
└── runs/                          # All experiment outputs (history, checkpoints, results)
```
---

## 4. How The Pipeline Works
 
### 4.1 Dataset Classes
 
**`MATWIVisionDataset`** (vision-only):
- Loads labels.csv, filters by split and set_range
- For each sample: opens the full 5496×3672 image → crops to the wear zone using 
  per-set coordinates from the paper's Table 2 (produces 600×400 px) → resizes to 
  target size (224×224 for ResNet50, 384×384 for EfficientNetV2) → normalises
- Returns: `{image, wear, wear_raw, set, image_id, type, material}`
 
**`MATWIMultimodalDataset`** (vision + sensors):
- Same image pipeline as above
- Additionally loads sensor CSVs and extracts 40 engineered features per sample:
  - 5 channels × 8 features = 40 total
  - Time domain (5 per channel): mean, std, rms, peak-to-peak, kurtosis
  - Frequency domain (3 per channel): dominant FFT frequency, low-band energy (0–200 Hz), 
    mid-band energy (200–800 Hz)
- Sensor features are standardised (zero mean, unit std) using a scaler fitted on 
  training data only
- Supports 3 feature subsets: `"all40"`, `"raw25"` (time-domain only), `"top25"` 
  (best by Ridge regression importance)
- Only keeps samples with BOTH modalities → 647 train / 300 val / 247 test
 
### 4.2 Models
 
**`MATWIVisionModel`** (vision-only):
- Backbone: ResNet50 (2048-dim) or EfficientNetV2-S (1280-dim) from timm, pretrained
- Head: `"simple"` (Linear(feat_dim, 1)) or `"mlp"` (Linear→ReLU→Dropout→Linear)
- Output: raw scalar prediction (no activation), normalised [0, 1]
- Also returns `image_embed` for future PINN loss
 
**`MATWIMultimodalModel`** (vision + sensors):
- Same EfficientNetV2-S backbone
- Three fusion modes:
  - **Early:** concatenate sensor features directly to 1280-dim backbone output → Linear head
  - **Intermediate:** sensor features through lightweight encoder (Linear→ReLU→Linear, 64-dim) 
    → concatenate with backbone → Linear head
  - **Late:** separate image and sensor prediction branches → learned 2→1 combiner
- All heads are simple Linear (no MLP) — critical finding from our ablation
 
### 4.3 Training
 
- Loss: L1 (MAE) or MSE — configurable per experiment
- Optimiser: AdamW (lr=3e-4, weight_decay=1e-4)
- Scheduler: fixed LR (paper replication) or OneCycleLR (our experiments)
- Gradient clipping: max_norm=1.0
- Best model saved by validation MAE
- Evaluation: MAE in µm, broken down by wear type (flank, adhesion, F+A)
 
---

## 5. Experiments & Results So Far
 
### 5.1 Vision-Only Ablation (9 experiments)
 
**Goal:** Replicate the paper's baseline and find the best vision-only setup.
 
**What we varied:** backbone (ResNet50 vs EfficientNetV2), normalisation (ImageNet vs 
dataset-specific), loss (L1 vs MSE), head type (simple vs MLP).
 
**What we kept fixed:** sets 1-13, 664 training images, no augmentation, LR 3e-4, 
17 epochs, batch size 32, AdamW.
 
**Results (test set MAE in µm, sorted by overall):**
 
| Rank | Experiment                           | Backbone | Head   | Norm     | Loss | Overall  | Flank | Adh  | F+A  |
|------|--------------------------------------|----------|--------|----------|------|----------|-------|------|------|
| 1    | efficientnetv2_dataset_MSE           | EffNetV2 | simple | dataset  | MSE  | **19.0** | 16.6  | 37.2 | 23.2 |
| 2    | resnet50_imagenet_L1                 | ResNet50 | simple | imagenet | L1   | **19.3** | 17.4  | 34.4 | 22.6 |
| 3    | efficientnetv2_imagenet_MSE          | EffNetV2 | simple | imagenet | MSE  | 22.7     | 20.0  | 39.3 | 28.9 |
| 4    | resnet50_imagenet_MSE                | ResNet50 | simple | imagenet | MSE  | 24.0     | 22.2  | 34.2 | 28.1 |
| 5    | efficientnetv2_imagenet_L1_mlp_sched | EffNetV2 | mlp    | imagenet | L1   | 26.0     | 23.2  | 42.1 | 32.8 |
| 6    | efficientnetv2_dataset_L1            | EffNetV2 | simple | dataset  | L1   | 26.1     | 17.6  | 37.0 | 58.3 |
| 7    | **Paper baseline**                   | ResNet50 | simple | ?        | ?    | **30.0** | 14.0  | 39.0 | 91.0 |
| 8    | efficientnetv2_imagenet_L1           | EffNetV2 | simple | imagenet | L1   | 30.1     | 20.1  | 34.6 | 70.6 |
| 9    | resnet50_dataset_L1                  | ResNet50 | simple | dataset  | L1   | 31.4     | 28.0  | 35.6 | 44.3 |
| 10   | resnet50_dataset_MSE                 | ResNet50 | simple | dataset  | MSE  | 33.1     | 29.8  | 34.2 | 46.7 |
 
**Key findings:**
1. **Simple head is critical.** Switching from MLP to simple Linear head was the single 
   biggest improvement (42→19 µm). On 664 samples, the MLP overfits badly.
2. **MSE loss crushes L1 on F+A.** Same backbone + norm, L1 gives 58.3 µm F+A while MSE 
   gives 23.2 µm. MSE penalises large errors quadratically, forcing the model to handle 
   hard adhesion cases.
3. **Dataset normalisation helps EfficientNetV2, ImageNet helps ResNet50.** Each backbone 
   works best with the normalisation closest to its pretraining distribution.
4. **We beat the paper.** Our best (19.0 µm) outperforms their regression baseline (30.0 µm) 
   and matches their histogram-loss approach (19 µm), which uses a fundamentally more complex 
   evaluation protocol with a reference image.
5. **F+A went from 91 µm (paper) to 23.2 µm (ours).** The simple head generalises far 
   better to unseen wear types than the MLP head the paper likely used.
 
### 5.2 Sensor Sanity Check (Ridge Regression)
 
**Goal:** Before building sensor fusion, check if the 40 engineered features carry 
predictive signal at all.
 
**Method:** Fit Ridge regression on standardised sensor features → wear (µm). No images.
 
**Results:**
 
| Method              | Test MAE | Train MAE | Notes                         |
|---------------------|----------|-----------|-------------------------------|
| Predict mean        | 40.9 µm  | —         | Baseline (always guess avg)   |
| Ridge (40 features) | 56.0 µm  | 21.7 µm   | WORSE than mean — overfitting |

**Feature importance (top 5):**
1. Fx_mid_energy (|coef|=116.5)
2. acc_mid_energy (|coef|=101.9)
3. Fx_low_energy (|coef|=81.4)
4. Fy_low_energy (|coef|=70.7)
5. Fx_mean (|coef|=62.1)
 
**Channel importance:** Fx (353) >> Fy (196) > acc (184) > acoustic (114) > Fz (103)
 
**Bottom features (essentially noise):** dom_freq (all channels), kurtosis (all channels), 
acc_mean, acc_low_energy.
 
**Verdict:** Sensor features have some structure but don't generalise across sets. The force 
channels carry most signal. Frequency-domain features (mid/low energy) matter more than 
time-domain statistics.
 
### 5.3 Sensor Fusion Ablation (10 experiments)
 
**Goal:** Controlled test — does adding sensors improve the best vision-only model?
 
**Setup:** All experiments identical to `efficientnetv2_dataset_MSE` except for sensor 
input. Same backbone, norm, loss, LR, epochs, batch size, seed. Uses the multimodal 
dataset class (647 train) for all runs including the vision-only control.
 
**Fusion architectures:**
- Early: concatenate sensor features to 1280-dim backbone → Linear(1280+N, 1)
- Intermediate: sensor encoder (Linear→ReLU→Linear, 64-dim) → concatenate → Linear(1344, 1)
- Late: separate branches → learned combiner
 
**Feature sets:**
- raw25: 25 time-domain features (mean, std, rms, p2p, kurtosis × 5 channels)
- top25: 25 Ridge-selected features (drops noise features, adds FFT energies)
- all40: all 40 features
 
**Results (test set MAE in µm):**
 
| Rank | Experiment           | Fusion       | Features | Overall | Flank | Adh   | F+A   | Epoch |
|------|----------------------|--------------|----------|---------|-------|-------|-------|-------|
| 1    | **vision_only_664**  | none         | —        | **19.0**| 16.6  | 37.2  | 23.2  | 14    |
| 2    | vision_only_647      | none         | —        | 22.4    | 20.3  | 35.1  | 27.1  | 15    |
| 3    | intermediate_top25   | intermediate | top25    | 29.1    | 27.6  | 54.2  | 26.6  | 3     |
| 4    | early_top25          | early        | top25    | 34.2    | 35.2  | 51.3  | **23.8** | 13 |
| 5    | intermediate_raw25   | intermediate | raw25    | 39.3    | 40.7  | 47.7  | 30.5  | 4     |
| 6    | late_raw25           | late         | raw25    | 53.6    | 54.3  | 50.4  | 52.0  | 3     |
| 7    | late_top25           | late         | top25    | 56.0    | 45.1  | 65.6  | 99.5  | 3     |
| 8    | late_all40           | late         | all40    | 68.8    | 55.8  | 75.4  | 122.5 | 4     |
| 9    | early_all40          | early        | all40    | 69.3    | 65.4  | 68.3  | 86.6  | 8     |
| 10   | intermediate_all40   | intermediate | all40    | 70.5    | 50.5  | 36.1  | 167.3 | 15    |
| 11   | early_raw25          | early        | raw25    | 75.6    | 70.4  | 115.3 | 84.5  | 2     |
 
**Key findings:**
1. **No sensor experiment beat vision-only.** Best sensor result (29.1 µm) is 6.7 µm 
   worse than the 647-sample control (22.4 µm).
2. **Intermediate fusion > Early > Late** — the lightweight encoder helps, but not enough.
3. **Top25 >> raw25 >> all40** — more features = worse performance. The 15 noise features 
   actively poison the model. Feature selection is essential.
4. **F+A shows a glimmer of hope.** Early_top25 gets 23.8 µm F+A, better than the 647 
   control (27.1 µm). Sensors carry some complementary signal for the hardest wear type, 
   but degrade everything else.
5. **Sensor experiments converge too early (epoch 2–4)** then overfit. The sensor patterns 
   are set-specific artifacts, not generalizable wear signals.
6. **The 647→664 gap costs 3.4 µm.** Losing 17 samples matters on this small dataset.
 
---

## 6. Current Conclusions
 
### What works:
- EfficientNetV2-S with a simple linear head, dataset normalisation, and MSE loss 
  achieves 19.0 µm — competitive with the paper's best (19 µm histogram-loss)
- The F+A gap (paper's 91 µm) is dramatically reduced to 23.2 µm just from the 
  simple head + MSE loss combination
- The vision backbone is already very strong on this dataset
 
### What doesn't work:
- Sensor features as currently engineered don't improve vision-only predictions
- The features overfit to set-specific patterns (proven by Ridge sanity check)
- More features = worse (all40 is catastrophic everywhere)
- Late fusion fails because sensors can't make standalone predictions
 
### What's next — Stage 3 (Taylor's Physics Loss):
The sensor fusion results actually strengthen the case for the physics-informed approach. 
Sensors failed because they're noisy and set-specific. Taylor's equation provides something 
fundamentally different — a physics-based prior on how wear *should* behave given the cutting 
conditions (speed, feed rate, material). This isn't additional noisy data; it's a mathematical 
constraint that regularises predictions.

The physics loss will penalise predictions that violate Taylor's tool-life equation 
(V · T^n = C), particularly when adhesion artifacts mislead the vision model. Unlike sensors, 
this constraint doesn't add learnable parameters or require additional data — it just reshapes 
the loss landscape to be physically consistent.
---
## 7. Note on Current Training Setup — Intentional Simplicity
All experiments so far use a deliberately simple and controlled setup: no data augmentation, 
no oversampling, no learning rate scheduling (for ResNet50 paper replication runs), a simple 
linear head, and fixed hyperparameters. This was chosen for two reasons:
 
1. **Paper comparability.** We needed to match the paper's setup as closely as possible to 
   validate our pipeline. The paper specifies very few details (ResNet50, LR 3e-4, 17 epochs, 
   664 images — that's it). By keeping our setup minimal, we reduce the number of unknowns 
   and can attribute performance differences to the variables we're actually testing (backbone, 
   loss, normalisation, sensors).
 
2. **Controlled ablation.** To answer "does X help?", every other variable must be held 
   constant. If we added augmentation AND sensors AND a new loss at the same time, we couldn't 
   tell which one caused any improvement or degradation.
 
**After completing Stage 3 (Taylor physics loss)** while keeping this setup, we plan to revisit all stages with 
additional optimisations. These are intentionally deferred, not forgotten.

---
## 8. List of Upcoming Tasks

### Stage 3 — Physics-Informed Loss (Taylor's Equation)
This is the core research contribution of the project. The key idea: add a physics 
penalty to the loss function based on Taylor's tool-life equation (V · T^n = C), 
so the model's predictions are constrained to be physically consistent.
 
- [ ] **Design the Taylor physics loss term.** Formulate how Taylor's equation translates 
  into a differentiable penalty. The loss becomes: `L_total = L_data (MSE) + λ · L_physics (Taylor residual)`. 
  Need to define what the Taylor residual looks like given our model's predictions and the 
  known cutting parameters (Vc, fz) per set.
- [ ] **Handle edge cases in the physics loss.** Set 1 has unknown cutting parameters 
  (cannot compute Taylor residual — exclude from physics loss). Set 6 has variable feed 
  rate mid-run (use with caution). Set 17 has z=2 inserts (different Taylor constant C).
- [ ] **Fit Taylor's equation constants (n, C) per material.** Need to determine the 
  material-dependent constants from the data. Fit separately for CK45 (sets 2–11) and 
  RVS 304 (sets 12–13). These become fixed inputs to the physics loss, not learned parameters.
- [ ] **Implement adaptive loss weighting (λ).** The balance between data loss and physics 
  loss matters — too much physics penalty might override the data signal, too little makes 
  it irrelevant. Options: fixed λ, linearly increasing λ over epochs, or learned λ.
- [ ] **Run controlled experiments.** Compare: vision-only (19.0 µm reference) vs 
  vision + Taylor loss vs vision + sensors + Taylor loss. Same setup as before — only the 
  loss function changes.
- [ ] **Analyse where physics helps.** The hypothesis is that Taylor's equation acts as a 
  regulariser specifically for adhesion cases where the visual appearance is misleading. 
  Track F+A MAE separately to test this.
- [ ] **Explore soft vs hard constraints.** Soft constraint = penalty in loss function 
  (Stage 3 main approach). Hard constraint = architectural enforcement (e.g., DeepONet 
  structure that makes violations mathematically impossible). This is RQ3.
 
### Stage 4 — Refinement & Optimisation
 
After the physics loss is working, there are several directions to explore. These can be 
divided across team members.
 
**Re-running earlier stages with better training settings:**
- [ ] **Data augmentation.** Horizontal flip, small rotation, brightness/contrast jitter, 
  Gaussian blur — we already have the code, just disabled. Test whether augmentation helps 
  on top of the physics loss.
- [ ] **Oversampling adhesion samples.** The weighted sampler (3× weight for adhesion types) 
  is implemented in the dataset class but was intentionally not used in controlled experiments. 
  Test it now.
- [ ] **Learning rate scheduling.** OneCycleLR was better for EfficientNetV2 in early tests 
  (26.0 µm mlp+sched vs 30.1 µm simple+fixed on imagenet L1). Re-test with the current best 
  setup.
- [ ] **More epochs / early stopping.** Some models converge at epoch 8, others at 15. 
  Implement patience-based early stopping to avoid overfitting and wasting compute.
- [ ] **Extended training set (sets 1–17).** Earlier multimodal experiments showed that 
  including sets 14–17 in training dramatically improved results (test MAE dropped from 
  44→24 µm for intermediate fusion). Re-test with the current best architecture.
 
**Model compression for edge deployment:**
- [ ] **Structured pruning.** Remove entire channels/filters that contribute least.
- [ ] **8-bit quantisation.** Reduce model size and inference time with minimal accuracy loss.
- [ ] **Knowledge distillation.** Train a smaller student model to mimic the full model.
- [ ] **Benchmark inference speed.** Measure latency on target edge hardware (if available) 
  or on CPU to simulate industrial deployment.
 
**Additional experiments (if time permits):**
- [ ] **Sensor feature engineering improvements.** The current 40 features are basic. Could 
  try: wavelet features, more FFT bands, sliding window statistics, learned features via 
  1D-CNN on raw sensor signals.
- [ ] **CNN-Transformer hybrid.** Replace or augment EfficientNetV2 with a Vision Transformer 
  or hybrid architecture to capture long-range spatial dependencies in wear patterns.
- [ ] **Per-set or per-material model heads.** Instead of one head for all sets, use 
  material-conditional prediction (separate final layers for CK45 vs RVS 304).
- [ ] **Cross-validation.** Current results are from a single train/val/test split. 
  Leave-one-set-out cross-validation would give more robust error estimates.
- [ ] **Uncertainty quantification.** MC Dropout or ensemble methods to provide confidence 
  intervals alongside point predictions — important for industrial deployment.

---
## 9. How to run stuff
First of all, make sure you have Snellius working by following the setup steps from [Wiki Snellius - Connecting to the system](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660216/Connecting+to+the+system). In this Wiki, you can also find info on how to write a job script, so please get familiar with this [Wiki Snellius - Writing a job script](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660220/Writing+a+job+script).

Once you are in and can use the terminal, you can create a folder in the `/scratch-shared/` folder by navigating to it using the following line:
```bash
cd /scratch-shared/your_username/
```

And then creating the folder by doing:
```bash
mkdir name_of_project_folder
```

Then, inside this folder, make sure to create the structure presented in Section 3. Also, ensure to have the environment setup, for this I would recommend asking Chat and it will explain it better, together with making sure you have all libraries needed installed.

 
### Vision ablation (already done):
```bash
cd /scratch-shared/your_username/name_of_project_folder/vision-only/
mkdir -p /scratch-shared/your_username/name_of_project_folder/runs/vision_ablation/logs
sbatch run_vision_ablation.sh
# After jobs finish:
python gather_results.py --output-dir /scratch-shared/your_username/name_of_project_folder/runs/vision_ablation
```
 
### Sensor fusion (already done):
```bash
cd /scratch-shared/your_username/name_of_project_folder/vision-sensor/
mkdir -p /scratch-shared/your_username/name_of_project_folder/runs/sensor_ablation/logs
sbatch run_sensor_ablation.sh
# After jobs finish:
python gather_results_sensor.py --output-dir /scratch-shared/your_username/name_of_project_folder/runs/sensor_ablation
```
 
### Check job status:
```bash
squeue -u your_username
```
 
### View results:
```bash
# Vision ablation:
cat /scratch-shared/your_username/name_of_project_folder/runs/vision_ablation/comparison_summary.csv
 
# Sensor ablation:
cat /scratch-shared/your_username/name_of_project_folder/runs/sensor_ablation/comparison_summary.csv
```