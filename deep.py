import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_classes,
                 pad_idx,
                 num_layers=1,
                 dropout_rate_emb=0.2,
                 dropout_rate_linear=0.3,
                 dropout_rate_hidden=0.1,
                 dropout_rate_inter_layer=0.1,
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm_layers = nn.ModuleList()
        self.inter_layer_dropouts = nn.ModuleList()

        for i in range(num_layers):
            input_dim = embed_dim if i == 0 else hidden_size
            self.lstm_layers.append(nn.Linear(input_dim + hidden_size, 4 * hidden_size))
            if i < num_layers - 1:
                self.inter_layer_dropouts.append(nn.Dropout(dropout_rate_inter_layer))

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

        for layer in self.lstm_layers:
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.zero_()
            layer.bias.data[self.hidden_size : 2 * self.hidden_size].fill_(1.0)
        nn.init.xavier_uniform_(self.V.weight)
        self.V.bias.data.zero_()

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = self.embed_dropout(embedded)

        batch_size, seq_len, _ = embedded.shape

        h_states = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c_states = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        for t in range(seq_len):
            layer_input = embedded[:, t, :]

            for i in range(self.num_layers):
                h_prev = h_states[i]
                c_prev = c_states[i]
                combined = torch.cat((layer_input, h_prev), dim=1)
                all_gates = self.lstm_layers[i](combined)
                ingate, forgetgate, cellgate, outgate = torch.chunk(all_gates, 4, dim=1)

                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = torch.tanh(cellgate)
                outgate = torch.sigmoid(outgate)

                c_new = (forgetgate * c_prev) + (ingate * cellgate)
                h_new = outgate * torch.tanh(c_new)

                h_states[i] = h_new
                c_states[i] = c_new

                if i < self.num_layers - 1:
                    layer_input = self.inter_layer_dropouts[i](h_new)
                else:
                    layer_input = h_new

        final_h = h_states[-1]
        final_h = self.output_dropout(final_h)

        output = self.V(final_h)
        output = self.linear_dropout(output)
        return output
