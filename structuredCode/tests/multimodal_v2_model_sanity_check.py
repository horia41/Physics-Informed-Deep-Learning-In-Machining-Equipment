from model.vision_sensor.multimodal_model_v2 import MATWIMultimodalModelV2
import torch

if __name__ == "__main__":
    print("=== Model architecture test ===\n")

    configs = [
        ("vision-only",        dict(use_sensors=False)),
        ("early-25",           dict(fusion_mode="early",        n_sensor_features=25)),
        ("intermediate-40",    dict(fusion_mode="intermediate", n_sensor_features=40)),
        ("late-25",            dict(fusion_mode="late",         n_sensor_features=25)),
        ("gated-25",           dict(fusion_mode="gated",        n_sensor_features=25)),
        ("gated-25 + mdrop",   dict(fusion_mode="gated",        n_sensor_features=25,
                                    modality_dropout_p=0.3)),
        ("intermediate + mdrop", dict(fusion_mode="intermediate", n_sensor_features=25,
                                      modality_dropout_p=0.3)),
    ]

    for label, kwargs in configs:
        model = MATWIMultimodalModelV2(**kwargs, pretrained=False)
        n_feat = kwargs.get("n_sensor_features", 40)
        use_s  = kwargs.get("use_sensors", True)
        images  = torch.randn(4, 3, 384, 384)
        sensors = torch.randn(4, n_feat) if use_s else None

        # train mode (exercises modality dropout)
        model.train()
        out_train = model(images, sensors)
        # eval mode
        model.eval()
        out_eval = model(images, sensors)

        gate_str = ("gate=%.3f" % out_eval["gate"].mean().item()
                    if out_eval["gate"] is not None else "gate=n/a")
        print(f"  {label:22s}: wear={tuple(out_eval['wear'].shape)}  {gate_str}")
        print()