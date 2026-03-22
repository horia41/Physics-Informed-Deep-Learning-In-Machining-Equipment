import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

class ResNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        
        in_features = self.model.fc.in_features   # this is 2048
        self.model.fc = nn.Linear(in_features, 1) # replace with single output
        
    def forward(self, x):
        
        return self.model(x).squeeze(1)  # shape [batch, 1] → [batch]


if __name__ == "__main__":

    model = ResNetRegressor()
    # test with a fake batch of 4 images
    fake_batch = torch.randn(4, 3, 224, 224)
    output = model(fake_batch)
    
    print(f"Input shape:  {fake_batch.shape}")   # [4, 3, 224, 224]
    print(f"Output shape: {output.shape}")       # [4]
    print(f"Output values: {output}") 

    summary(model, (3, 224, 224))