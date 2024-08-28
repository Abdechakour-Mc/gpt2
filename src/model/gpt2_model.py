import torch
import torch.nn as nn

from transformer_block import TransformerBlock, PositionalEncoding



class GPT2Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        super(GPT2Embedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(token_emb)
        return self.dropout(pos_emb)


class GPT2Model(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, ff_h_size, max_len, dropout=0.1):
        super(GPT2Model, self).__init__()

        self.embedding = GPT2Embedding(vocab_size, max_len, dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_size, heads, ff_h_size, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)

        # Pass through each transformer block
        for block in self.transformer_blocks:
            x = block(x, x, x, mask)

        # Apply the final layer
        logits = self.fc_out(x)

        return logits