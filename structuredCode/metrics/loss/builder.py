import torch.nn as nn

def make_criterion(loss_name: str) -> nn.Module:
    if loss_name == "L1":
        return nn.L1Loss()
    elif loss_name == "MSE":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}. Use 'L1' or 'MSE'.")