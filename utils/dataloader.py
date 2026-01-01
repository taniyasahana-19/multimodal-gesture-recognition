
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from utils.augmentations import train_transform, test_transform

def get_dataloaders(path, img_size, channels, batch_size=32, split=True):
    full = ImageFolder(path, transform=train_transform(img_size, channels))
    if split:
        train_len = int(0.8 * len(full))
        val_len = len(full) - train_len
        train_ds, val_ds = random_split(full, [train_len, val_len])
        val_ds.dataset.transform = test_transform(img_size, channels)
    else:
        train_ds = full
        val_ds = ImageFolder(path.replace("train","test"),
                             transform=test_transform(img_size, channels))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    )
