
import torch.nn as nn
from torchvision import models

def build(num_classes, channels):
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    if channels != 3:
        model.features[0] = nn.Conv2d(channels,64,3,1,1)
    model.classifier = nn.Sequential(
        nn.Linear(25088,512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512,num_classes)
    )
    return model
