import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score


def compute_f1(y_true, y_pred):
    assert y_true.ndim == 2
    assert y_true.shape == y_pred.shape

    return f1_score(y_true, y_pred, average='macro')


def train(epochs, model, optim, criterion, train_loader, val_loader, val_gt, scheduler=None):
    for epoch in range(epochs):
        print(f'Eposh:\t{epoch+1}')
        model.train()
        train_loss = 0.0
        for texts, labels in tqdm(train_loader):
            optim.zero_grad()
            out, _ = model(texts)
            loss = criterion(out, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for texts, labels in tqdm(val_loader):
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
        if scheduler is not None:
            scheduler.step(f1)

        current_lr = optim.param_groups[0]['lr']

        print()
        print(f"│ Train: {train_loss/len(train_loader):.2f} │ Val: {val_loss/len(val_loader):.2f} │ F1: {f1*100:.2f} │ LR: {current_lr:.6f} │")

