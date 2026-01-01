
import torch.nn as nn
from torchvision import models

def build(num_classes, channels):
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
    if channels != 3:
        model.Conv2d_1a_3x3.conv = nn.Conv2d(channels,32,3,2,0,bias=False)
    model.fc = nn.Sequential(
        nn.Linear(2048,512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512,num_classes)
    )
    return model
