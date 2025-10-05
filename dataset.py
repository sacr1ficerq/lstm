import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, device):
        self.data = [torch.tensor(vocab.txt2id(txt), dtype=torch.long, device=device) for txt in texts]
        self.labels = [torch.tensor(l, dtype=torch.float32, device=device) for l in labels]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def collate(batch):
    texts, labels = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels_tensor = torch.stack(labels)
    return padded_texts, labels_tensor
