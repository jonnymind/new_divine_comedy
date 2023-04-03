#!/bin/env python
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from encoder import *

from trainer import Trainer
from bigram_model import BigramLanguageModel


# Meta parameters
torch.manual_seed(1337)
block_size = 16
batch_size = 8
eval_interval = 100
eval_iters = 200
train_iters = 100
learning_rate = 3e-3
n_embed = 32
n_heads = 4
n_layers = 4
dropout = 0.2
#---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

# Dictionary Loading
print("Loading dictionaries")
dictionary = load_tokenized_vector("commedia_tokens.json")
dict_size = len(dictionary)
print(f"Working with a dictionary size of {dict_size}")
train_data = torch.tensor(load_tokenized_vector("commedia_training_set.json"), dtype=torch.long)
val_data = torch.tensor(load_tokenized_vector("commedia_validation_set.json"), dtype=torch.long)
enc = Encoder(dictionary)

# Model generation
model = BigramLanguageModel(dict_size, block_size, n_embed, n_heads, n_layers, dropout)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
trainer = Trainer(model, optimizer, train_data, val_data)
trainer(train_iters, batch_size, block_size)

# Example output generation
idx0 = torch.zeros((1,1), dtype=torch.long)
idx0.to(device)
prevision = model.generate(idx0, max_new_tokens=500)
print("---- Result ----")
print(enc.decode(prevision[0].tolist()))
