#!/bin/env python
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from encoder import *
import os

from trainer import Trainer
from bigram_model import BigramLanguageModel
import sys
import io
import functools

print = functools.partial(print, flush=True)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Meta parameters
torch.manual_seed(1337)
block_size = 256
batch_size = 32
eval_interval = 100
loss_samples = 1000
train_iters = 5000
save_iters = 100
learning_rate = 3e-4
n_embed = 384
n_heads = 6
n_layers = 6
dropout = 0.2
#---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

dict, input, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
# Dictionary Loading
print("Loading dictionaries")
dictionary = load_tokenized_vector(dict)
dict_size = len(dictionary)
print(f"Working with a dictionary size of {dict_size}")
data = load_tokenized_vector(input)
print(f"Loaded data of {len(data)} tokens.")
val_split = ((len(data) // 5) * 4)
train_data = torch.tensor(data[0: val_split], dtype=torch.long)
print(f"Train data length {len(train_data)} tokens.")
val_data = torch.tensor(data[val_split:], dtype=torch.long)
print(f"Validation data length {len(val_data)} tokens.")
enc = Encoder(dictionary)

# Model generation
if os.path.isfile(model_name):
    print(f"Loading existing {model_name}")
    model = torch.load(model_name)
    print(f"dict_size: {model.voc_size}")
else:
    model = BigramLanguageModel(dict_size, block_size, n_embed, n_heads, n_layers, dropout)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
trainer = Trainer(model, optimizer, train_data, val_data, eval_interval, loss_samples, model_name, save_iters)
trainer(train_iters, batch_size, block_size)

print(f"Saving model to {model_name}")
torch.save(model, model_name)

