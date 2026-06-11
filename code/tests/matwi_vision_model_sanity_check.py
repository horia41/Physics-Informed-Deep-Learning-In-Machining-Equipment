import torch
from constants.vision_only_constants import BACKBONES
from model.vision_only.vision_model import MATWIVisionModel

if __name__ == "__main__":
    for bb in ["resnet50", "efficientnetv2_s"]:
        for ht in ["simple", "mlp"]:
            info = BACKBONES[bb]
            model = MATWIVisionModel(backbone=bb, head_type=ht, pretrained=False)
            x = torch.randn(2, 3, info["default_size"], info["default_size"])
            out = model(x)
            print(f"  {bb}/{ht}: wear={out['wear'].shape}, "
                  f"embed={out['image_embed'].shape}\n")