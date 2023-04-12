#!/bin/env python
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from encoder import *

from bigram_model import BigramLanguageModel
import sys
import io
import functools

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#---
device = "cuda" if torch.cuda.is_available() else "cpu"
print = functools.partial(print, flush=True)
print(f"Using device {device}")

dict, model_file, token_count = sys.argv[1], sys.argv[2], int(sys.argv[3])
if len(sys.argv) > 4:
    torch.manual_seed(int(sys.argv[4]))

# Dictionary Loading
print("Loading dictionaries")
dictionary = load_tokenized_vector(dict)
dict_size = len(dictionary)
print(f"Working with a dictionary size of {dict_size}")
enc = Encoder(dictionary)

# Data generation
print(f"Loading model {model_file}")
model = torch.load(model_file)

idx0 = torch.zeros((1,1), dtype=torch.long)
idx0.to(device)
prevision = model.generate(idx0, max_new_tokens=token_count)
print()
print("---- Result ----")
print(enc.decode(prevision[0].tolist()))
