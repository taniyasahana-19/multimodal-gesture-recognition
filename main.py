
import argparse, torch
from utils.config import DATASETS, IMAGE_SIZES
from utils.dataloader import get_dataloaders
from training.trainer import train
from training.evaluator import evaluate
from training.plots import plot
from models import mobilenetv2, vgg19, inceptionv3

parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--model")
parser.add_argument("--mode", choices=["rgb","depth","rgbd"])
args = parser.parse_args()

channels = {"rgb":3,"depth":1,"rgbd":4}[args.mode]
classes = DATASETS[args.dataset]["classes"]
img_size = IMAGE_SIZES[args.model]

train_loader, val_loader = get_dataloaders(
    f"datasets/{args.dataset}/{args.mode}",
    img_size, channels,
    split=(args.dataset!="OUHANDS")
)

builder = {"mobilenetv2":mobilenetv2,
           "vgg19":vgg19,
           "inceptionv3":inceptionv3}[args.model]

model = builder.build(classes, channels)
device = "cuda" if torch.cuda.is_available() else "cpu"

history = train(model, train_loader, val_loader, 25, device)
plot(history)
print(evaluate(model, val_loader, device))
