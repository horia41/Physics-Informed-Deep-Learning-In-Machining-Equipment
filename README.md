# Physics Informed Deep Learning In Machining Equipment (PINNs)

## Some updates 
- folder `dataset` handles everything in terms of data ; has `download_data.py` which automatically downloads the MATWI dataset and creates a folder `dataset/matwi` where it will be downloaded; has `EDA.py` which will output in a folder `dataset/eda_output` and performs exploratory data analysis on our dataset which will help us see details about our data and establish what data preprocessing steps and feature engineering we should perform; here, still missing the DataLoader part
- created folders `vision-baseline`, `vision-sensor` and `pinn` where we'll develop each model accordingly so we dont necessarily have to create multiple branches while using the same `dataset` folder

So, for now, make sure to run the `dataset/download_data.py` and then the `dataset/EDA.py` so you can see the plots and outputs of it.


## Another thing - Dataset Classes

### DatasetClass_Vision.py — `MATWIVisionDataset`
 
Vision-only dataset for Vision Only baseline. Returns cropped, normalised images with wear labels.
 
### Instantiation
 
```python
dataset = MATWIVisionDataset(
    data_dir         = "./data/matwi",   # root folder with SetX subfolders
    labels_csv       = "./data/matwi/labels.csv",
    sets_csv         = "./data/matwi/sets.csv",
    split            = "train",          # "train" | "val" | "test" | "unseen" | "all"
    set_range        = "1-13",           # "1-13" (paper) | "1-17" (extended)
    normalisation    = "dataset",        # "dataset" | "imagenet"
    augment          = False,            # True only affects split="train"
    augment_strategy = "uniform",        # "uniform" | "oversample_adhesion"
    wear_cap         = 450.0,            # cap wear in µm, None to disable
    impute_zero_wear = False,            # True = fill missing wear with 0.0
    image_size       = (224, 224),       # resize target for EfficientNetV2
)
```
 
### What each option does
 
| Option             | Effect                                                                                                                                   |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `split`            | Selects sets: train={1,2,5,7,8,10,11}, val={3,6,12}, test={4,9,13}, unseen={14–17}                                                       |
| `set_range`        | `"1-13"` replicates the paper exactly. `"1-17"` adds sets 14–17 to training (val/test unchanged)                                         |
| `normalisation`    | `"dataset"` uses precomputed stats for the selected `set_range`. `"imagenet"` uses ImageNet stats                                        |
| `augment`          | Applies horizontal flip, ±5° rotation, brightness/contrast jitter, mild blur. No vertical flip or hue shifts (would destroy wear signal) |
| `augment_strategy` | `"oversample_adhesion"` requires using `dataset.get_sampler()` with DataLoader — oversamples adhesion cases 3×                           |
| `wear_cap`         | Clips wear at 450 µm (handles the Set 11 outlier at 750 µm). Set to `None` to keep raw values                                            |
| `impute_zero_wear` | `False` drops the 140 rows with missing wear (recommended for Stage 1). `True` fills with 0.0 (useful for PINN stage)                    |
 
### Each batch returns
 
```python
{
    "image":    tensor (3, 224, 224),  # cropped, normalised
    "wear":     tensor scalar,          # normalised [0, 1] (wear_µm / 1000)
    "wear_raw": tensor scalar,          # original µm value, for MAE reporting
    "set":      int,
    "image_id": int,
    "type":     str,                    # "flank_wear" | "adhesion" | "flank_wear+adhesion"
    "material": str,                    # "CK45" | "RVS 304"
}
```
 
### Sample counts (set_range="1-13", wear_cap=450, impute_zero_wear=False)
 
| Split | n | Notes |
|---|---|---|
| train | 664 | Matches paper exactly. No `flank_wear+adhesion` in training sets |
| val | 301 | 123 `flank_wear+adhesion` samples — hard validation set |
| test | 254 | 46 `flank_wear+adhesion` samples |
 
With `set_range="1-17"`, training grows to **1108** samples and gains **166** `flank_wear+adhesion` samples from Sets 16–17.
 
### Quick start
 
```python
loaders = build_vision_dataloaders(
    data_dir      = "./data/matwi",
    labels_csv    = "./data/matwi/labels.csv",
    sets_csv      = "./data/matwi/sets.csv",
    set_range     = "1-13",
    augment_train = False,
    batch_size    = 32,
    num_workers   = 4,
)
for batch in loaders["train"]:
    images = batch["image"]    # (B, 3, 224, 224)
    wear   = batch["wear"]     # (B,) normalised
```
 
---
 
### DatasetClass_VisionSensors.py — `MATWIMultimodalDataset`
 
Vision + sensor dataset for Vision+sensor part. Extends the vision class with 40 engineered sensor features per sample. Only rows with both image and sensor files present are included.
 
### Sensor features (40 total)
 
For each of the 5 sensor channels (`acc`, `acoustic`, `Fx`, `Fy`, `Fz`), 8 features are extracted from the raw sensor CSV (~99,000 rows per file, sampled at 0.6 ms):
 
| Feature           | Type      | Description                                |
|-------------------|-----------|--------------------------------------------|
| `mean`            | time      | Signal mean                                |
| `std`             | time      | Signal standard deviation                  |
| `rms`             | time      | Root mean square                           |
| `p2p`             | time      | Peak-to-peak amplitude                     |
| `kurtosis`        | time      | Signal spikiness (high = impulsive events) |
| `dom_freq`        | frequency | Dominant FFT frequency (Hz), DC excluded   |
| `low_band_energy` | frequency | FFT power in 0–200 Hz band                 |
| `mid_band_energy` | frequency | FFT power in 200–800 Hz band               |
 
> Note on Force Z: non-zero in practice (mean ≈ −2.32, std ≈ 2.09) despite the paper's claim. It is retained and handled by standardisation rather than dropped.
 
### Sensor standardisation — important workflow
 
Features have wildly different scales (`dom_freq` ≈ hundreds vs `acc_mean` ≈ 0). A `SensorScaler` (zero mean, unit std) must be fit on training data only and passed to val/test datasets.
 
```python
# 1. Build training dataset — no scaler yet
train_ds = MATWIMultimodalDataset(
    ..., split="train", sensor_scaler=None
)
 
# 2. Fit scaler on training data only
scaler = train_ds.fit_sensor_scaler(save_path="sensor_scaler.pkl")
 
# 3. Pass fitted scaler to val and test — never refit
val_ds  = MATWIMultimodalDataset(..., split="val",  sensor_scaler=scaler)
test_ds = MATWIMultimodalDataset(..., split="test", sensor_scaler=scaler)
 
# 4. Reload scaler later without refitting
scaler = SensorScaler.load("sensor_scaler.pkl")
```
 
> Never call `fit_sensor_scaler()` on val or test — leaks statistics.
 
### Each batch returns
 
```python
{
    "image":           tensor (3, 224, 224),  # same as vision class
    "sensor_features": tensor (40,),           # standardised sensor features
    "wear":            tensor scalar,           # normalised [0, 1]
    "wear_raw":        tensor scalar,           # µm
    "set":             int,
    "image_id":        int,
    "type":            str,
    "material":        str,
}
```
 
### Sample counts vs vision class
 
| Split        | Vision (n) | Multimodal (n) | Difference                   |
|--------------|------------|----------------|------------------------------|
| train (1–13) | 664        | 647            | −17 rows lost to sync errors |
| val          | 301        | 300            | −1                           |
| test         | 254        | 254            | 0                            |
 
### Quick start
 
```python
 
loaders, scaler = build_multimodal_dataloaders(
    data_dir         = "./data/matwi",
    labels_csv       = "./data/matwi/labels.csv",
    sets_csv         = "./data/matwi/sets.csv",
    set_range        = "1-13",
    batch_size       = 32,
    num_workers      = 4,
    scaler_save_path = "sensor_scaler.pkl",
)
for batch in loaders["train"]:
    images  = batch["image"]            # (B, 3, 224, 224)
    sensors = batch["sensor_features"]  # (B, 40)
    wear    = batch["wear"]             # (B,)
```
 
---
 
### Normalisation stats reference
 
| set_range    | mean (RGB)               | std (RGB)                |
|--------------|--------------------------|--------------------------|
| `"1-13"`     | [0.4056, 0.3962, 0.5084] | [0.1980, 0.1920, 0.2173] |
| `"1-17"`     | [0.4290, 0.4174, 0.5292] | [0.2002, 0.1936, 0.2142] |
| `"imagenet"` | [0.485, 0.456, 0.406]    | [0.229, 0.224, 0.225]    |
 
---
 
### Train/val/test split reference
 
| Split  | Sets                  | Material      | Notes                                                 |
|--------|-----------------------|---------------|-------------------------------------------------------|
| train  | 1, 2, 5, 7, 8, 10, 11 | CK45          | No `flank_wear+adhesion` samples                      |
| val    | 3, 6, 12              | CK45 + RVS304 | Hard split — 41% adhesion                             |
| test   | 4, 9, 13              | CK45 + RVS304 | 24% adhesion                                          |
| unseen | 14, 15, 16, 17        | RVS304 only   | 52% adhesion — added to train when `set_range="1-17"` |



### Experiments that need to be performed

| ID      | Sets | Norm     | Augment | Modality   | wear_cap          | Purpose                                                                                                                                |
|:--------|:-----|:---------|:--------|:-----------|:------------------|:---------------------------------------------------------------------------------------------------------------------------------------|
| **E01** | 1–13 | imagenet | No      | Image only | 450 $\mu\text{m}$ | **Paper replica — direct comparison to ResNet50 baseline (MAE $\approx$ 30 $\mu\text{m}$ total)**<br>Goal: match paper numbers exactly |
| **E02** | 1–13 | dataset  | No      | Image only | 450 $\mu\text{m}$ | **Same as E01 but with dataset-specific normalisation**<br>Goal: measure effect of normalisation choice                                |
| **E03** | 1–13 | dataset  | Yes     | Image only | 450 $\mu\text{m}$ | **Add safe augmentations (flip, rotation, jitter, blur)**<br>Goal: measure effect of augmentation on 664 training images               |
| **E04** | 1–13 | dataset  | Yes     | Image only | 450 $\mu\text{m}$ | **E03 + oversample adhesion $3\times$ via WeightedRandomSampler**<br>Goal: measure effect of adhesion oversampling on val adhesion MAE |
| **E05** | 1–17 | dataset  | No      | Image only | 450 $\mu\text{m}$ | **Add sets 14–17 to training (444 extra samples, 166 flank+adhesion)**<br>Goal: measure effect of extra adhesion training data         |
| **E06** | 1–17 | dataset  | Yes     | Image only | 450 $\mu\text{m}$ | **Sets 1–17 + augmentation — best Stage 1 configuration**<br>Goal: establish best vision-only ceiling before adding sensors            |