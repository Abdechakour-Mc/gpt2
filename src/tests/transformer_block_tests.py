import torch
from ..model.transformer_block import PositionalEncoding, TransformerBlock
from ..model.gpt2_model import GPT2Model

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

def test_gpt2_model():
    vocab_size = 30522  # Example vocab size
    embed_size = 512
    num_layers = 6
    heads = 8
    ff_hidden_size = 2048
    max_len = 100
    seq_length = 10
    batch_size = 2

    # Create random input tensor simulating token indices (batch_size, seq_length)
    input_data = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Initialize GPT-2 model
    gpt2_model = GPT2Model(vocab_size, embed_size, num_layers, heads, ff_hidden_size, max_len)

    # Apply the GPT-2 model
    logits = gpt2_model(input_data)

    print("GPT-2 model output shape:", logits.shape)



# Run the test
test_transformer_block()
test_gpt2_model()
