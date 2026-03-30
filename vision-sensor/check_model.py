"""
Model sanity check — test_model.py
====================================
Verifies model.py output shapes and forward passes for all fusion modes.
Run with: python test_model.py

No real data needed — uses random tensors matching dataset output shapes.
"""

import torch
from modelVisionSensor import MATWIWearModel, N_SENSOR_FEATURES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}\n")

B = 4
dummy_images  = torch.randn(B, 3, 384, 384).to(device)
dummy_sensors = torch.randn(B, N_SENSOR_FEATURES).to(device)

# ── All three fusion modes with sensors ───────────────────────────────────────
for fusion in ["early", "intermediate", "late"]:
    print(f"\n{'─'*60}")
    print(f"fusion_mode='{fusion}' | use_sensors=True")
    print(f"{'─'*60}")

    model = MATWIWearModel(
        fusion_mode = fusion,
        use_sensors = True,
        pretrained  = False,
    ).to(device)

    out = model(dummy_images, dummy_sensors)

    print(f"  wear shape        : {out['wear'].shape}")
    print(f"  wear sample       : {out['wear'].detach().squeeze()}")
    print(f"  image_embed shape : {out['image_embed'].shape}")
    if out["sensor_embed"] is not None:
        print(f"  sensor_embed shape: {out['sensor_embed'].shape}")
    if out["pred_image"] is not None:
        print(f"  pred_image shape  : {out['pred_image'].shape}")
        print(f"  pred_sensor shape : {out['pred_sensor'].shape}")

# ── Image only (no sensors) ───────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("fusion_mode='intermediate' | use_sensors=False (image only)")
print(f"{'─'*60}")

model_vision = MATWIWearModel(
    fusion_mode = "intermediate",
    use_sensors = False,
    pretrained  = False,
).to(device)

out = model_vision(dummy_images, sensor_features=None)
print(f"  wear shape  : {out['wear'].shape}")
print(f"  wear sample : {out['wear'].detach().squeeze()}")

print("\nAll checks passed.\n")