from nltk.sem.drt import DrtParser
from torch import dropout
import torch.nn as nn


class TorchRNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_classes,
                 pad_idx,
                 dropout_rate_emb=0.2,
                 dropout_rate_linear=0.3,
                 dropout_rate_hidden=0.1,
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Use PyTorch's built-in RNN
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            batch_first=True,
            nonlinearity='tanh'
        )

        self.output_layer1 = nn.Linear(hidden_size, num_classes)

        self.embed_dropout = nn.Dropout(dropout_rate_emb)
        self.output_dropout = nn.Dropout(dropout_rate_hidden)
        self.linear_dropout = nn.Dropout(dropout_rate_linear)

        # self.relu = nn.ReLU()

        self.pad_idx = pad_idx

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.weight.data[self.pad_idx].zero_()

        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                param.data.zero_()

        nn.init.xavier_uniform_(self.output_layer1.weight)
        # nn.init.xavier_uniform_(self.output_layer2.weight)

        self.output_layer1.bias.data.zero_()
        # self.output_layer2.bias.data.zero_()

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = self.embed_dropout(embedded)

        rnn_output, _ = self.rnn(embedded)
        # rnn_output shape: (batch_size, seq_len, hidden_size)
        # hidden shape: (1, batch_size, hidden_size)

        last_hidden = rnn_output[:, -1, :]  # (batch_size, hidden_size)

        last_hidden = self.output_dropout(last_hidden)

        output = self.output_layer1(last_hidden)
        output = self.linear_dropout(output)
        # output = self.relu(output)
        # output = self.output_layer2(output)

        return output
