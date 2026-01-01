
import torch
from sklearn.metrics import classification_report

def evaluate(model, loader, device):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device)
            out=model(x)
            y_true.extend(y.numpy())
            y_pred.extend(out.argmax(1).cpu().numpy())
    return classification_report(y_true, y_pred, digits=4)
