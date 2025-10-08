import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score


def compute_f1(y_true, y_pred):
    assert y_true.ndim == 2
    assert y_true.shape == y_pred.shape

    return f1_score(y_true, y_pred, average='macro')


def train(epochs, model, optim, criterion, train_loader, val_loader, val_gt, 
          warmup_scheduler, plateau_scheduler, warmup_epochs):
    global_step = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        is_warmup_phase = epoch < warmup_epochs

        for texts, labels in train_loader:
            optim.zero_grad()
            out = model(texts)
            loss = criterion(out, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item()
            if is_warmup_phase:
                warmup_scheduler.step()
            global_step += 1

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for texts, labels in val_loader:
                out = model(texts)
                val_loss += criterion(out, labels).item()

        probs_batches = []
        with torch.no_grad():
            for texts, labels in val_loader:
                out = model(texts)
                probs = torch.sigmoid(out).cpu()
                probs_batches.append(probs)

        pred = (torch.cat(probs_batches, dim=0) >= 0.5).numpy().astype(np.int32)
        f1 = compute_f1(val_gt, pred)
        current_lr = optim.param_groups[0]['lr']
        if not is_warmup_phase:
            plateau_scheduler.step(f1)

        print(f"│ Epoch {epoch+1 if epoch >= 9 else '0' + str(epoch+1)} │ Train Loss: {train_loss/len(train_loader):.4f} │ Val Loss: {val_loss/len(val_loader):.4f} │ F1: {f1*100:.2f}% │ LR: {current_lr:.8f} │")
