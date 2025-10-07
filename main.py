from sklearn.model_selection import train_test_split

import torch
from torch.optim import AdamW, lr_scheduler
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from rnn import RNN
from torchrnn import TorchRNN
from vocab import Vocab
from dataset import TextDataset, collate, load_biotech
from train import train

EPOCHS = 10
LR = 1e-4
WD = 1e-5

FACTOR = 0.5
PATIENCE = 3

config = {
    'embed_dim': 64,
    'hidden_size': 128,
    'num_classes': 28,
    'dropout_rate': 0.3
}

print('Loading dataset...')
dataset, labels = load_biotech()

device = 'cuda:0'
texts_train, texts_test, y_train, y_test = train_test_split(
    dataset.text.values,
    dataset.encoded_labels.values,
    test_size=0.2,  # do not change this
    random_state=0  # do not change this
)

print('Building vocab...')
vocab = Vocab(texts_train, no_below=4, no_above=0.7)
print('Vocab size:', len(vocab.dictionary))

print('Tokenizing ...')
train_dataset = TextDataset(texts_train, y_train, vocab, device)
test_dataset = TextDataset(texts_test, y_test, vocab, device)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=lambda batch: collate(batch, vocab.pad_idx)
)

val_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=lambda batch: collate(batch, vocab.pad_idx)
)

# class_weights = torch.tensor((y_train.shape[0] - y_train.sum(0)) / (y_train.sum(0) + 1e-6), dtype=torch.float32, device=device)
class_weights = torch.tensor((y_train.shape[0]) / (y_train.sum(0) + 1e-6), dtype=torch.float32, device=device)

config['vocab_size'] = len(vocab.dictionary)
config['pad_idx'] = vocab.pad_idx

gt = torch.stack([torch.tensor(label) for label in y_test], dim=0).cpu()
model = TorchRNN(**config).to(device)

train_config = {
    'epochs': EPOCHS,
    'model': model,
    'optim': AdamW(model.parameters(), lr=LR, weight_decay=WD),
    'criterion': BCEWithLogitsLoss(pos_weight=class_weights),
    'train_loader': train_loader,
    'val_loader': val_loader,
    'val_gt': gt,
}

train_config['scheduler'] = lr_scheduler.ReduceLROnPlateau(
        train_config['optim'],
        mode='max',
        factor=FACTOR,
        patience=PATIENCE
)

print('Training model...')
train(**train_config)
