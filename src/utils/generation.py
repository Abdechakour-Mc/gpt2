import torch

def generate_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def generate(model, idx, max_new_tokens, context_size, temperature, top_k):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Using only the last token
        logits = logits[:, -1, :]

    # Top K
        top_logits, top_pos = torch.topk(logits, 2)
        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(-1, top_pos, top_logits)
    
    # Temperature
        if temperature > 0.0:
            scaled_logits = masked_logits / temperature
            probas = torch.softmax(scaled_logits, dim=-1)
        else:
            probas = torch.softmax(masked_logits, dim=-1)


    # Probabilistic Sampling
        idx_next = torch.multinomial(probas, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def text_to_token_ids(text, tokenizer):
    token_ids = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    return token_ids

def token_ids_to_text(token_ids, tokenizer):
    flat_token_ids = token_ids.squeeze(0)
    return tokenizer.decode(flat_token_ids.tolist())
