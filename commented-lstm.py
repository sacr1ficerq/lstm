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

        # In an LSTM, we have 4 gates (input, forget, cell, output).
        # We can create one large linear layer for efficiency.
        # It takes the concatenated input and previous hidden state.
        self.gates = nn.Linear(embed_dim + hidden_size, 4 * hidden_size)

        # Output layer remains the same
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

        # Initialize biases to zero, except for the forget gate's bias.
        self.gates.bias.data.zero_()
        self.V.bias.data.zero_()
        # A common LSTM trick: initialize the forget gate's bias to 1.
        # This encourages the model to remember information by default.
        # The bias vector is for [input_gate, forget_gate, cell_gate, output_gate].
        # We set the bias for the forget_gate (the second chunk) to 1.
        self.gates.bias.data[self.hidden_size : 2 * self.hidden_size].fill_(1.0)


    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = self.embed_dropout(embedded)

        batch_size, seq_len, embed_dim = embedded.shape

        # Initialize both hidden state and cell state to zeros
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Process the sequence one step at a time
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # (batch_size, embed_dim)

            # Concatenate input and previous hidden state
            combined = torch.cat((x_t, h), dim=1) # (batch_size, embed_dim + hidden_size)

            # Pass through the single gate layer
            # all_gates shape: (batch_size, 4 * hidden_size)
            all_gates = self.gates(combined)

            # Split the gates into 4 parts
            # Each part has shape (batch_size, hidden_size)
            i, f, g, o = torch.chunk(all_gates, 4, dim=1)

            # Apply activation functions
            # i_t: Input gate (what new information to store)
            i = torch.sigmoid(i)
            # f_t: Forget gate (what to forget from the cell state)
            f = torch.sigmoid(f)
            # g_t: Cell gate / candidate (new candidate values)
            g = torch.tanh(g)
            # o_t: Output gate (what to output from the cell state)
            o = torch.sigmoid(o)

            c = f * c + i * g

            h = o * torch.tanh(c)
            h = self.output_dropout(h)

        output = self.V(h)
        output = self.linear_dropout(output)
        return output
