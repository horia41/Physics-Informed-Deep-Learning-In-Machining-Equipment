from dataclasses import dataclass

@dataclass
class HardConfig:
    name:          str
    backbone:      str   = "efficientnetv2_s"
    set_range:     str   = "1-13"
    image_size:    tuple = (384, 384)
    epochs:        int   = 17
    lr:            float = 3e-4
    normalisation: str   = "dataset"
    weight_decay:  float = 1e-4
    batch_size:    int   = 32
    use_scheduler: bool  = True
    data_loss:     str   = "mse"
