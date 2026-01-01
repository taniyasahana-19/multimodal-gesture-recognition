
import torch
from tqdm import tqdm

def train(model, train_loader, val_loader, epochs, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    history = {"train_acc":[], "val_acc":[]}

    model.to(device)
    for e in range(epochs):
        model.train()
        correct=total=0
        for x,y in tqdm(train_loader):
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad()
            out=model(x)
            loss=criterion(out,y)
            loss.backward()
            optimizer.step()
            correct+=(out.argmax(1)==y).sum().item()
            total+=y.size(0)
        history["train_acc"].append(correct/total)

        model.eval()
        correct=total=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device)
                out=model(x)
                correct+=(out.argmax(1)==y).sum().item()
                total+=y.size(0)
        history["val_acc"].append(correct/total)

    return history
