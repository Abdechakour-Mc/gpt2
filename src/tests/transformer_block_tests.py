import torch
from ..model.transformer_block import PositionalEncoding, SelfAttention

def test_transformer_blocks():
    embed_size = 512
    heads = 8
    seq_length = 10
    batch_size = 2

    # Create random input tensor simulating token embeddings (batch_size, seq_length, embed_size)
    input_data = torch.randn(batch_size, seq_length, embed_size)

    # Positional encoding
    pos_encoder = PositionalEncoding(embed_size)
    encoded_data = pos_encoder(input_data)
    
    # Self-attention
    self_attention = SelfAttention(embed_size, heads)
    
    # Simulating values, keys, queries (input data itself in transformer models)
    mask = None
    output = self_attention(encoded_data, encoded_data, encoded_data, mask)
    
    print("Self-attention output shape:", output.shape)

# Run the test
test_transformer_blocks()