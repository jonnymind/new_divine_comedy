#!/bin/env python
import sys
import json
from encoder import *

def tokenize_text(text):
    voc = {}
    tokens = []
    for char in text:
        if char not in voc:
            voc[char] = len(voc)
            tokens.append(char)
    return sorted(tokens)

srcfile, dictfile, outfile = sys.argv[1], sys.argv[2], sys.argv[3]
print(f"Input data {srcfile}; tokens written to {dictfile}; tokenized output written to {outfile}")

with open(srcfile,"r", encoding="utf-8") as fin:
    text = fin.read()

voc = tokenize_text(text)
voc_size = len(voc)

print(f"Found dictionary of {voc_size} tokens")
with open(dictfile, "w") as f:
    json.dump(voc, f)

encoder = Encoder(voc)
result = encoder.encode(text)
print(f"Writing tokenized output of {len(result)} entries")
with open(outfile, "w") as f:
    json.dump(result, f)
