import torch
from model.vision_sensor.multimodal_v1 import MATWIMultimodalModelV1

if __name__ == "__main__":
    print("=== Model architecture test ===\n")

    configs = [
        ("vision-only",   dict(use_sensors=False)),
        ("early-25",      dict(fusion_mode="early",        n_sensor_features=25)),
        ("early-40",      dict(fusion_mode="early",        n_sensor_features=40)),
        ("intermediate-25", dict(fusion_mode="intermediate", n_sensor_features=25)),
        ("intermediate-40", dict(fusion_mode="intermediate", n_sensor_features=40)),
        ("late-25",       dict(fusion_mode="late",          n_sensor_features=25)),
        ("late-40",       dict(fusion_mode="late",          n_sensor_features=40)),
    ]

    for label, kwargs in configs:
        model = MATWIMultimodalModelV1(**kwargs, pretrained=False)
        images  = torch.randn(2, 3, 384, 384)
        sensors = torch.randn(2, kwargs.get("n_sensor_features", 40)) if kwargs.get("use_sensors", True) else None
        out = model(images, sensors)
        print(f"  {label:20s}: wear={out['wear'].shape}, "
              f"embed={out['image_embed'].shape}")
        print()
