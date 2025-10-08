import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from tqdm import tqdm

num_workers = 4


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, device):
        self.data = [torch.tensor(vocab.txt2id(txt), dtype=torch.long, device=device) for txt in tqdm(texts)]
        self.labels = [torch.tensor(l, dtype=torch.float32, device=device) for l in tqdm(labels)]

    @staticmethod
    def _process_text(text, vocab):
        return vocab.txt2id(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def collate(batch, pad_idx):
    texts, labels = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=pad_idx)
    labels_tensor = torch.stack(labels)
    return padded_texts, labels_tensor


def load_preprocessed():
    dataset = pd.read_csv('data/preprocessed.csv')

    all_labels = set()
    for labels_str in dataset.labels:
        all_labels.update(labels_str.split(', '))

    keys = sorted(all_labels - {'other'})
    label_to_idx = {label: idx for idx, label in enumerate(keys)}

    def encode_labels(labels_str):
        if labels_str == 'other':
            return np.zeros(len(keys), dtype=np.float32)

        encoded = np.zeros(len(keys), dtype=np.float32)
        for label in labels_str.split(', '):
            if label in label_to_idx:
                encoded[label_to_idx[label]] = 1.0
        return encoded

    dataset['encoded_labels'] = [encode_labels(labels) for labels in dataset.labels]
    return dataset, keys


def load_biotech():
    dataset = pd.read_csv('data/biotech_news.tsv', sep='\t')

    all_labels = set()
    for labels_str in dataset.labels:
        all_labels.update(labels_str.split(', '))

    keys = sorted(all_labels - {'other'})
    label_to_idx = {label: idx for idx, label in enumerate(keys)}

    def encode_labels(labels_str):
        if labels_str == 'other':
            return np.zeros(len(keys), dtype=np.float32)

        encoded = np.zeros(len(keys), dtype=np.float32)
        for label in labels_str.split(', '):
            if label in label_to_idx:
                encoded[label_to_idx[label]] = 1.0
        return encoded

    dataset['encoded_labels'] = [encode_labels(labels) for labels in dataset.labels]

    return dataset, keys
