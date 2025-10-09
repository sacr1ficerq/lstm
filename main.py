from sklearn.model_selection import train_test_split

import torch
from torch.optim import AdamW, lr_scheduler
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

import numpy as np

from rnn import RNN
from deep import LSTM
from torchrnn import TorchRNN
# from vocab import Vocab
from vocab_bpe import BPEVocab
from dataset import TextDataset, collate, load_biotech, load_preprocessed
from train import train

BATCH_SIZE = 256

WARMUP_EPOCHS = 12
EPOCHS = 72
LR = 3e-3
WD = 1e-3

FACTOR = 0.5
PATIENCE = 5

config = {
    'embed_dim': 128,
    'hidden_size': 256,
    'num_classes': 28,
    'dropout_rate_emb': 0.1,
    'dropout_rate_linear': 0.5,
    'dropout_rate_hidden': 0.1,
    'num_layers': 2
}

print('Loading dataset...')
# dataset, labels = load_biotech()
dataset, labels = load_preprocessed()

device = 'cuda:0'
texts_train, texts_test, y_train, y_test = train_test_split(
    dataset.text.values,
    dataset.encoded_labels.values,
    test_size=0.2,  # do not change this
    random_state=0  # do not change this
)

print('Building vocab...')
# vocab = Vocab(texts_train, no_below=26, no_above=0.68, max_tokens=36, clean=False)
vocab = BPEVocab(texts_train, vocab_size=4000, max_tokens=100, clean=False)

# print('Vocab size:', len(vocab.dictionary))

print('Tokenizing ...')
train_dataset = TextDataset(texts_train, y_train, vocab, device)
test_dataset = TextDataset(texts_test, y_test, vocab, device)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda batch: collate(batch, vocab.pad_idx)
)

val_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=lambda batch: collate(batch, vocab.pad_idx)
)

class_weights = torch.tensor((y_train.shape[0] - y_train.sum(0)) / (y_train.sum(0) + 1e-6), dtype=torch.float32, device=device) # type: ignore
# class_weights = torch.tensor((y_train.shape[0]) / (y_train.sum(0) + 1e-6), dtype=torch.float32, device=device)

config['vocab_size'] = vocab.tokenizer.vocab_size()
config['pad_idx'] = vocab.pad_idx

gt = torch.stack([torch.tensor(label) for label in y_test], dim=0).cpu()
model = LSTM(**config).to(device)

train_config = {
    'warmup_epochs': WARMUP_EPOCHS,
    'epochs': EPOCHS,
    'model': model,
    'optim': AdamW(model.parameters(), lr=LR, weight_decay=WD),
    'criterion': BCEWithLogitsLoss(pos_weight=class_weights),
    'train_loader': train_loader,
    'val_loader': val_loader,
    'val_gt': gt,
}


total_warmup_steps = WARMUP_EPOCHS * len(train_loader) 

warmup_scheduler = lr_scheduler.LinearLR(
    train_config['optim'],
    start_factor=0.01,
    end_factor=1.0,
    total_iters=total_warmup_steps
)


plateau_scheduler = lr_scheduler.ReduceLROnPlateau(
    train_config['optim'],
    mode='max',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
)

train_config['warmup_scheduler'] = warmup_scheduler
train_config['plateau_scheduler'] = plateau_scheduler

print('Training model...')
train(**train_config)
