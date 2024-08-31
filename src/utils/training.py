import torch.nn as nn

class GPT2Loss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(GPT2Loss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, target_ids):
        """
        :param logits: Tensor of shape (batch_size, sequence_length, vocab_size)
        :param target_ids: Tensor of shape (batch_size, sequence_length)
        :return: Cross-Entropy Loss
        """

        logits = logits.view(-1, logits.size(-1))
        target_ids = target_ids.view(-1)

        loss = self.loss_fn(logits, target_ids)
        return loss


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)

        # Forward pass
        logits = model(input_ids)

        # Compute loss
        loss_fn = GPT2Loss(ignore_index=-100)
        loss = loss_fn(logits, target_ids)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

