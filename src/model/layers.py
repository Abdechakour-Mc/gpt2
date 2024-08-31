# Custom layers like feed-forward, layer norm, etc.
import torch.nn as nn

class ResidualConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.layer_norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        # Apply the sublayer (like self-attention or feed-forward) than add residual connection
        return x + self.dropout(sublayer(self.layer_norm(x)))
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_size, ff_h_size=2048, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_h_size)
        self.fc2 = nn.Linear(ff_h_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))