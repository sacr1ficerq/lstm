import torch
import torch.nn as nn

class LSTM(nn.Module):
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

        self.gates = nn.Linear(embed_dim + hidden_size, 4 * hidden_size)

        self.V = nn.Linear(hidden_size, num_classes)

        self.embed_dropout = nn.Dropout(dropout_rate_emb)
        self.output_dropout = nn.Dropout(dropout_rate_hidden)
        self.linear_dropout = nn.Dropout(dropout_rate_linear)

        self.pad_idx = pad_idx
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.weight.data[self.pad_idx].zero_()

        nn.init.xavier_uniform_(self.gates.weight)
        nn.init.xavier_uniform_(self.V.weight)

        self.gates.bias.data.zero_()
        self.V.bias.data.zero_()
        self.gates.bias.data[self.hidden_size : 2 * self.hidden_size].fill_(1.0)


    def forward(self, x):
        embedded = self.embedding(x)  
        embedded = self.embed_dropout(embedded)

        batch_size, seq_len, embed_dim = embedded.shape

        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            x_t = embedded[:, t, :]  

            combined = torch.cat((x_t, h), dim=1) 

            all_gates = self.gates(combined)

            i, f, g, o = torch.chunk(all_gates, 4, dim=1)

            i = torch.sigmoid(i)
            
            f = torch.sigmoid(f)
            
            g = torch.tanh(g)
            
            o = torch.sigmoid(o)

            c = f * c + i * g

            h = o * torch.tanh(c)
            h = self.output_dropout(h)

        output = self.V(h)
        output = self.linear_dropout(output)
        return output
