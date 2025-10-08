import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # --- Attention Layers ---
        # W_a in the paper: transforms the LSTM hidden states.
        self.W_attention = nn.Linear(hidden_size, hidden_size, bias=False)
        # v_a in the paper: a learned vector to score the transformed states.
        self.v_attention = nn.Linear(hidden_size, 1, bias=False)

        # The final projection layer now takes the context vector as input
        self.V = nn.Linear(hidden_size, num_classes)

        self.embed_dropout = nn.Dropout(dropout_rate_emb)
        self.output_dropout = nn.Dropout(dropout_rate_hidden) # We can apply this to the context vector
        self.linear_dropout = nn.Dropout(dropout_rate_linear)

        self.pad_idx = pad_idx
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.weight.data[self.pad_idx].zero_()

        nn.init.xavier_uniform_(self.gates.weight)
        self.gates.bias.data.zero_()
        self.gates.bias.data[self.hidden_size : 2 * self.hidden_size].fill_(1.0)

        # Initialize attention weights
        nn.init.xavier_uniform_(self.W_attention.weight)
        nn.init.xavier_uniform_(self.v_attention.weight)

        nn.init.xavier_uniform_(self.V.weight)
        self.V.bias.data.zero_()

    def forward(self, x, return_attention=False):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = self.embed_dropout(embedded)

        batch_size, seq_len, embed_dim = embedded.shape

        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # We need to store all hidden states from the LSTM
        all_hidden_states = []

        for t in range(seq_len):
            x_t = embedded[:, t, :]
            combined = torch.cat((x_t, h), dim=1)
            all_gates = self.gates(combined)
            i, f, g, o = torch.chunk(all_gates, 4, dim=1)

            i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)

            c = f * c + i * g
            h = o * torch.tanh(c)

            all_hidden_states.append(h)

        # Stack all hidden states to get a tensor
        # lstm_outputs shape: (batch_size, seq_len, hidden_size)
        lstm_outputs = torch.stack(all_hidden_states, dim=1)

        # --- Attention Mechanism ---

        # 1. Calculate alignment scores (energy)
        # We pass all LSTM outputs through a linear layer and a tanh activation
        # energy shape: (batch_size, seq_len, hidden_size)
        energy_pre = self.W_attention(lstm_outputs)
        energy_tanh = torch.tanh(energy_pre)

        # Multiply by the learned vector v_a to get a score per timestep
        # energy shape: (batch_size, seq_len, 1)
        energy = self.v_attention(energy_tanh)

        # 2. Compute attention weights using softmax
        # We must mask the padding tokens so they get an attention weight of 0.
        mask = (x != self.pad_idx).unsqueeze(2) # (batch_size, seq_len, 1)
        energy.masked_fill_(mask == 0, -1e10) # Replace padding with a very small number

        # attention_weights shape: (batch_size, seq_len, 1)
        attention_weights = F.softmax(energy, dim=1)

        # 3. Compute the context vector
        # This is the weighted sum of the LSTM hidden states.
        # context_vector shape: (batch_size, hidden_size)
        context_vector = torch.sum(attention_weights * lstm_outputs, dim=1)

        # Apply dropout to the context vector
        context_vector = self.output_dropout(context_vector)

        # 4. Final Prediction
        # Use the context vector for the final classification
        output = self.V(context_vector)
        output = self.linear_dropout(output)

        if return_attention:
            return output, attention_weights.squeeze(2) # Squeeze for easier visualization
        else:
            return output
