import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.W = nn.Linear(embed_dim, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x, h=None):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        batch_size, seq_len, embed_dim = embedded.shape
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # (batch_size, embed_dim)
            h = self.tanh(self.W(x_t) + self.U(h))
        
        output = self.V(h)
        return output, h
