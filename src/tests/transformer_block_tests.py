import torch
from ..model.transformer_block import PositionalEncoding, TransformerBlock


def test_transformer_block():
    embed_size = 512
    heads = 8
    seq_length = 10
    batch_size = 2
    ff_hidden_size = 2048

     # Create random input tensor simulating token embeddings (batch_size, seq_length, embed_size)
    input_data = torch.randn(batch_size, seq_length, embed_size)

    # Positional encoding
    pos_encoder = PositionalEncoding(embed_size)
    encoded_data = pos_encoder(input_data)
    
    assert not torch.all(encoded_data == input_data), "Nothing changed"

    # Initialize Transformer block
    transformer_block = TransformerBlock(embed_size, heads, ff_hidden_size)
    
    # Apply the Transformer block (assuming no masking is needed for this test)
    output = transformer_block(encoded_data, encoded_data, encoded_data, mask=None)
    
    assert not torch.all(encoded_data == output), "Nothing changed"

    print("Transformer block output shape:", output.shape)

# Run the test
test_transformer_block()
