
```markdown
# GPT-2 From Scratch

A PyTorch implementation of the GPT-2 language model from scratch, including tokenizer, model architecture, training utilities, and text generation capabilities.

## Features

- Complete GPT-2 model implementation with:
  - Multi-head self-attention
  - Positional embeddings
  - Transformer blocks
  - Residual connections
- Byte Pair Encoding (BPE) tokenizer
- Training pipeline with:
  - Custom dataset loader
  - Training loop
  - Loss calculation
- Text generation with:
  - Greedy decoding
  - Temperature sampling
  - Top-k sampling

## Directory Structure

```
abdechakour-mc-gpt2/
├── README.md
└── src/
    ├── model/
    │   ├── gpt2_model.py        # Main GPT-2 model implementation
    │   ├── layers.py            # Custom layers (FFN, Residual)
    │   └── transformer_block.py # Transformer block with self-attention
    ├── scripts/
    │   └── preprocess_data.py   # Data preprocessing utilities
    ├── tests/
    │   ├── data_loader_test.py  # Data loading tests
    │   ├── training_test.py    # Training loop tests
    │   └── transformer_block_tests.py # Model component tests
    ├── tokenizer/
    │   └── tokenizer.py         # BPE Tokenizer implementation
    └── utils/
        ├── data_loader.py       # Dataset and DataLoader
        ├── generation.py        # Text generation utilities
        └── training.py          # Training utilities
```

## Requirements

- Python 3.7+
- PyTorch
- tiktoken (for tokenization)
- Other standard libraries (re, os, math, collections)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/abdechakour-mc-gpt2.git
cd abdechakour-mc-gpt2
```

2. Install dependencies:
```bash
pip install torch tiktoken
```

## Usage

### Training

```python
from src.model.gpt2_model import GPT2Model
from src.utils.data_loader import create_dataloader
from src.utils.training import train_one_epoch
import torch.optim as optim

# Initialize model
model = GPT2Model(vocab_size=50257, embed_size=768, num_layers=12, 
                 heads=12, ff_h_size=3072, max_len=1024, bias=True)

# Create dataloader
dataloader = create_dataloader("path/to/text/files", batch_size=4, max_len=256)

# Set up optimizer
optimizer = optim.AdamW(model.parameters(), lr=3e-5)

# Train
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
avg_loss = train_one_epoch(model, dataloader, optimizer, device)
```

### Text Generation

```python
from src.utils.generation import generate, text_to_token_ids, token_ids_to_text
import tiktoken

# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Prepare input
prompt = "The quick brown fox"
input_ids = text_to_token_ids(prompt, tokenizer)

# Generate text
generated_ids = generate(model, input_ids, max_new_tokens=50, 
                        context_size=256, temperature=0.7, top_k=50)

# Decode to text
generated_text = token_ids_to_text(generated_ids, tokenizer)
print(generated_text)
```

## Testing

Run the included tests to verify model components:

```bash
python -m src.tests.transformer_block_tests
python -m src.tests.training_test
python -m src.tests.data_loader_test
```

## Customization

Key parameters you can adjust:

- Model size:
  - `embed_size`: Embedding dimension (768 for base GPT-2)
  - `num_layers`: Number of transformer blocks (12 for base GPT-2)
  - `heads`: Number of attention heads (12 for base GPT-2)
  - `ff_h_size`: Feed-forward hidden size (3072 for base GPT-2)

- Training:
  - `max_len`: Context window size
  - `batch_size`: Training batch size
  - `stride`: Sliding window stride for training data

- Generation:
  - `temperature`: Controls randomness (0.0-1.0)
  - `top_k`: Limits sampling to top k tokens

## License

[MIT License](LICENSE)

## Acknowledgements

This implementation was inspired by the original GPT-2 paper:
"Language Models are Unsupervised Multitask Learners" by OpenAI
```
