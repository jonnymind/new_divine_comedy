#!/bin/env python
from encoder import *

with open("commedia.txt", "r") as dataFile:
    text = dataFile.read()
tokens = load_tokenized_vector("commedia_tokens.json")
encoder = Encoder(tokens)

result = encoder.encode(text)
save_tokenized_vector(result, "commedia_data.json")
