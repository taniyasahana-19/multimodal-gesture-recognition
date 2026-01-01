
import torch.nn as nn
from torchvision import models

def build(num_classes, channels):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    if channels != 3:
        model.features[0][0] = nn.Conv2d(channels,32,3,2,1,bias=False)
    model.classifier = nn.Sequential(
        nn.Linear(model.last_channel,512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512,num_classes)
    )
    return model
