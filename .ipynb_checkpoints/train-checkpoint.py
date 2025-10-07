import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

def compute_f1(y_true, y_pred):
    assert y_true.ndim == 2
    assert y_true.shape == y_pred.shape

    return f1_score(y_true, y_pred, average='macro')

def train(epochs, model, optim, criterion, train_loader, val_loader, val_gt):
    print('started training')
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for texts, labels in train_loader:
            optim.zero_grad()
            out, _ = model(texts)
            loss = criterion(out, labels)
            loss.backward()
            optim.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for texts, labels in val_loader:
                out, _ = model(texts)
                val_loss += criterion(out, labels).item()

        probs_batches = []
        with torch.no_grad():
            for texts, labels in val_loader:
                out, _ = model(texts)
                probs = torch.sigmoid(out).cpu()
                probs_batches.append(probs)

        pred = (torch.cat(probs_batches, dim=0) >= 0.5).numpy().astype(np.int32)
        f1 = compute_f1(val_gt, pred)

        print()
        print(f"┌───────────┬──────────────┬────────────┬─────────────┐")
        print(f"│ Epoch {epoch:3d} │ Train: {train_loss/len(train_loader):.4f} │ Val: {val_loss/len(val_loader):.4f} │ F1: {f1:.4f} │")
        print(f"└───────────┴──────────────┴────────────┴─────────────┘")

