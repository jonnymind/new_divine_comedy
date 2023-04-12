#!/bin/env python
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from encoder import *

from trainer import Trainer
from bigram_model import BigramLanguageModel

import sys
import io
import functools
# Create a new output stream with UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Meta parameters
torch.manual_seed(1337)
block_size = 256
batch_size = 32
eval_iters = 100
estimate_iters = 1000
train_iters = 5000
learning_rate = 3e-4
n_embed = 384
n_heads = 6
n_layers = 6
dropout = 0.2
#---
device = "cuda" if torch.cuda.is_available() else "cpu"
print = functools.partial(print, flush=True)
print(f"Using device {device}")

def tokenize_file(filename):
    voc = {}
    rvoc = {}
    tokens = []
    with open(filename,"r", encoding="utf-8") as fin:
        text = fin.read()
        for char in text:
            if char not in voc:
                token = len(voc)
                voc[char] = token
                rvoc[token] = char
            else:
                token = voc[char]
            tokens.append(token)
    return rvoc, tokens


def decoder(voc, data):
    text = ""
    for token in data:
        text += voc[token]
    return text

# Dictionary Loading
print("Creating dictionaries")
voc, tokenized_data = tokenize_file("commedia.txt")
voc_size = len(voc)
print(f"Working with a dictionary size of {voc_size}")
train_data = torch.tensor(tokenized_data[:len(tokenized_data)//4*3], dtype=torch.long)
val_data = torch.tensor(tokenized_data[:-len(tokenized_data)//4*3], dtype=torch.long)

# Model generation
model = BigramLanguageModel(voc_size, block_size, n_embed, n_heads, n_layers, dropout)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
trainer = Trainer(model, optimizer, train_data, val_data, eval_interval=eval_iters, estimate_iters=estimate_iters)
trainer(train_iters, batch_size, block_size)
torch.save(model.state_dict(), 'model_state_dict.pt')

# Example output generation
idx0 = torch.zeros((1,1), dtype=torch.long)
prevision = model.generate(idx0, max_new_tokens=500)
print("---- Result ----")
print(decoder(voc, prevision[0].tolist()))
