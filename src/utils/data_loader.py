# Handling the dataset
import os
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader



class GPTDataset(Dataset):
    def __init__(self, data_dir, max_len, stride):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.data_dir = data_dir
        self.input_ids = []
        self.target_ids = []
        

        
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path ,'r', encoding='utf-8') as f:
                    text = f.read()

                # Tokenize the text
                tokenized_ids = self.tokenizer.encode(text)
        assert tokenized_ids, "Check file path"

        for i in range(0, len(tokenized_ids)- max_len, stride):
            input_chunk = tokenized_ids[i:i + max_len]
            target_chunk = tokenized_ids[i + 1: i + max_len + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
    def __len__(self):
        return len(self.input_ids)
    

def create_dataloader(data_dir, batch_size=4, max_len=256, stride=128, shuffle=True, drop_last=True):
    dataset = GPTDataset(data_dir, max_len, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader