import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, pad_idx, dropout_rate=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Input gate weights
        self.W_ii = nn.Linear(embed_dim, hidden_size, bias=True)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=False)

        # Forget gate weights
        self.W_if = nn.Linear(embed_dim, hidden_size, bias=True)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=False)

        # Cell gate weights
        self.W_ig = nn.Linear(embed_dim, hidden_size, bias=True)
        self.W_hg = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output gate weights
        self.W_io = nn.Linear(embed_dim, hidden_size, bias=True)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output layer
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

        # Initialize all LSTM weights with Xavier uniform
        for layer in [self.W_ii, self.W_hi, self.W_if, self.W_hf, 
                     self.W_ig, self.W_hg, self.W_io, self.W_ho]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.zero_()

        nn.init.xavier_uniform_(self.V.weight)
        self.V.bias.data.zero_()

    def forward(self, x, hc=None):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = self.embed_dropout(embedded)

        batch_size, seq_len, embed_dim = embedded.shape

        # Initialize hidden state and cell state if not provided
        if hc is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h, c = hc

        for t in range(seq_len):
            x_t = embedded[:, t, :]  # (batch_size, embed_dim)

            # LSTM gates
            i_t = torch.sigmoid(self.W_ii(x_t) + self.W_hi(h))  # Input gate
            f_t = torch.sigmoid(self.W_if(x_t) + self.W_hf(h))  # Forget gate
            g_t = torch.tanh(self.W_ig(x_t) + self.W_hg(h))     # Cell gate
            o_t = torch.sigmoid(self.W_io(x_t) + self.W_ho(h))  # Output gate

            # Update cell state and hidden state
            c = f_t * c + i_t * g_t
            h = o_t * torch.tanh(c)

            # Apply dropout to hidden state for next time step
            if t < seq_len - 1:  # Don't apply to the last hidden state before output
                h = self.hidden_dropout(h)

        # Apply dropout to final hidden state before output layer
        h = self.output_dropout(h)
        output = self.V(h)
        return output, (h, c)
