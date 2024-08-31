import torch

import tiktoken
from ..model.gpt2_model import GPT2Model
from ..utils.data_loader import create_dataloader
from ..utils.training import train_one_epoch


def test_training_loop():
    # Hyperparameters and settings
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab  # Example vocabulary size
    embed_size = 512
    num_layers = 6
    heads = 8
    ff_hidden_size = 2048
    max_len = 256
    batch_size = 4
    stride = 128
    learning_rate = 3e-5

    # Initialize model, optimizer, and dataloader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2Model(vocab_size, embed_size, num_layers, heads, ff_hidden_size, max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    data_dir = "."
    dataloader = create_dataloader(data_dir, batch_size=batch_size, max_len=max_len, stride=stride)

    # Train for one epoch
    avg_loss = train_one_epoch(model, dataloader, optimizer, device)
    
    print(f"Average training loss: {avg_loss:.4f}")

# Run the test
test_training_loop()
