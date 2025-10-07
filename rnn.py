import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, pad_idx, dropout_rate=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.W = nn.Linear(embed_dim, hidden_size, bias=True)
        self.U = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, num_classes)

        self.embed_dropout = nn.Dropout(dropout_rate)
        self.hidden_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

        self.pad_idx = pad_idx
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # set padding embedding to zero
        self.embedding.weight.data[self.pad_idx].zero_()

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.V.weight)

        self.W.bias.data.zero_()
        self.V.bias.data.zero_()

    def forward(self, x, h=None):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = self.embed_dropout(embedded)

        batch_size, seq_len, embed_dim = embedded.shape
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            x_t = embedded[:, t, :]  # (batch_size, embed_dim)
            h = torch.tanh(self.W(x_t) + self.U(h))

            # Apply dropout to hidden state for next time step
            if t < seq_len - 1:  # Don't apply to the last hidden state before output
                h = self.hidden_dropout(h)

        # Apply dropout to final hidden state before output layer
        h = self.output_dropout(h)
        output = self.V(h)
        return output, h

        output = self.V(h)
        return output, h
