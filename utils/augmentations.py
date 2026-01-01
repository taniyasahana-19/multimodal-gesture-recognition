
from torchvision import transforms

def train_transform(size, channels):
    mean = [0.5]*channels
    std = [0.5]*channels
    return transforms.Compose([
        transforms.Resize((size,size)),
        transforms.RandomRotation(40),
        transforms.RandomAffine(0, translate=(0.2,0.2), shear=20, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def test_transform(size, channels):
    mean = [0.5]*channels
    std = [0.5]*channels
    return transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
