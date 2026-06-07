# EDA Takeways
1. SAMPLING MISMATCH <br/>
    Each sensor CSV = ~tens of thousands of rows for a single image measurement.<br/>
    → Need feature engineering (FFT, RMS, Kurtosis) to create per-pass summary vectors.

2. SYNC ERRORS<br/>
    Some rows have image but no sensor (or vice versa).<br/>
    → Dataset class must handle missing modalities gracefully.

3. ADHESION IS THE HARD CASE<br/>
    Regression MAE jumps from ~14 µm (flank) to ~91 µm (flank+adhesion) in the paper.<br/>
    → Physics constraint (Taylor) should act as a corrective prior here.

4. SET 17 HAS z=2 (TWO INSERTS)<br/>
    Taylor's equation uses z (number of teeth). Set 17 behaves differently.<br/>
    → Either fit Taylor's C separately, or treat Set 17 as a separate domain.

5. SET 1 HAS UNKNOWN CUTTING PARAMETERS<br/>
    Cannot apply Taylor's equation to Set 1 without imputing Vc.<br/>
    → Either exclude from PINN loss or impute from similar sets (2–4 range).

6. SET 6 HAS VARIABLE Vf<br/>
    Feed rate changed mid-run → Taylor assumption of constant V is violated.<br/>
    → Use with caution in physics loss; consider excluding from Taylor computation.

7. TRAIN/VAL/TEST SPLIT IS SET-LEVEL (no data leakage)<br/>
    Never mix rows from the same set across splits.<br/>
    → Always split at the Set level, not at the sample level.

8. RVS 304 (Sets 12–17) PRODUCES MORE ADHESION<br/>
    Including them in training helps generalisation but makes the task harder.<br/>
    → Paper excluded Sets 14–17 entirely; worth experimenting with inclusion.

# NORM STATS FOR IMAGES:

## WHEN USING SETS 1-13:
split=train | sets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] | n=1219 | wear_cap=450.0µm<br/>
mean: [0.40560478076171874, 0.39624402888997395, 0.5083500896809896]<br/>
std: [0.19804957525771552, 0.1920124143538162, 0.2173112347091954]
-------------------------------------------------------------------------------------------------------------------------------------


## WHEN USING SETS 1-17:
split=train | sets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] | n=1663 | wear_cap=450.0µm<br/>
mean: [0.42896144165039063, 0.4173878169759115, 0.5292330250651042]<br/>
std: [0.20020134730149317, 0.19364826735868443, 0.21420715969306942]
-------------------------------------------------------------------------------------------------------------------------------------

