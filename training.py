#!/bin/env python
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from encoder import *

torch.manual_seed(1337)
block_size = 256
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")
eval_interval = 100
eval_iters = 200
train_iters = 3000
learning_rate = 3e-4
n_embed = 384
n_heads = 6
n_layers = 6
dropout = 0.2


def get_batch(data):
    split_pos = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack( [data[pos : pos+block_size] for pos in split_pos])
    y = torch.stack( [data[pos+1 : pos+block_size+1] for pos in split_pos])
    x, y = x.to(device), y.to(device)
    return x, y

class BigramLanguageModel(nn.Module):

    def __init__(self, voc_size):
        super().__init__()
        # In our case, the embeddings are the probability distribution of each next-token (column)
        # given the current token (row). The sum-total the value in each row must be 1.
        # Call it next-token probability distribution (NTPD) matrix

        # could be (voc_size, voc_size), but in practice we can ignore the less commonly following tokens.
        self.token_embedding_table = nn.Embedding(voc_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # ... but to do that we need a index-map so that the numbers in each row xi don't represent the
        # probability of the i-th token to appear after y-i, but an index in an header table.
        self.blocks = nn.Sequential( *[Block(n_embed, n_heads) for _ in range(n_layers)],
            nn.LayerNorm(n_embed)
        )
        self.sa_head = nn.Linear(n_embed, voc_size)

    def forward(self, inputs, targets = None):
        Btc, Blk = inputs.shape
        # Input and targets are both (batch_size, block_size)
        # This gives us a matrix with every token in every batch replaced by the respective NTPD
        tok_emb = self.token_embedding_table(inputs)  # (Bch, Blk, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(Blk, device=device)) # (Blk, n_embed)
        x = tok_emb + pos_emb
        blocks_out = self.blocks(x) # (Bch, Blk, n_embeds)
        logits = self.sa_head(blocks_out) # (Bch, Blk, Voc)

        if targets == None:
            loss = None
        else:
            Bch, Blk, Voc = logits.shape
            # Cross-entropy wants a vector or a 2D matrix, we need to flatten the batch dimension.
            entropy_logits = logits.view(Bch * Blk, Voc)
            targets = targets.view(Bch * Blk)
            # Cross-entropy is a good and fast measure of the difference between two prob-dists.
            loss = F.cross_entropy(entropy_logits, targets)
        return logits, loss
    
    def generate(self, inputs, max_new_tokens):
        # Inputs should be in (Btc, Blk)
        for _ in range(max_new_tokens):
            idx_context = inputs[:, -block_size: ]
            logits, loss = self(idx_context) 
            # Get the NTPD for the last token in each block.
            last_logits = logits[:, -1, :]  # (Btc, Voc)
            # Turn their predictions in a distribution 
            probs = F.softmax(last_logits, dim=-1)  # (Btc, Voc)
            # Extracts one token index according with the probability of happening next
            idx_next = torch.multinomial(probs, num_samples=1)  # (Btc)
            # Add the extracted tokens to the existing dataset
            inputs = torch.cat((inputs, idx_next), dim=1) # (Btc)
        return inputs

def train(train_data, val_data, model, optimizer):
    for step in range(train_iters):
        if step % eval_interval == 0:
            losses = estimate_loss(train_data, val_data, model)
            print(f"Step {step}: train loss={losses['train']}; validation loss={losses['val']}")

        xb, yb = get_batch(train_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

class SelfAttHead(nn.Module):
    """ one head of self attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # Apply the linear models to the input
        k = self.key(x)
        q = self.query(x)
        # Cross the two applied models, transposing the time and channel
        wei = q @ k.transpose(-2, -1)  # (B, T, C) @ (B, C, T) -> (B, C, T)
        # Normalize Variance
        wei = wei * C ** -0.5
        # Prevent influence of future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Compute distribution
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # Weighed aggregation of the values.
        v = self.value(x)
        out = wei @ v
        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    

@torch.no_grad()
def estimate_loss(train_data, val_data, model):
    out = {}
    model.eval()
    for type, split in [("train",train_data), ("val",val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[type] = losses.mean()
    model.train()
    return out

class FeedForward(nn.Module):
    """Moves the token weights one step forward"""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Trasformer block: communication followed by computation"""
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 

# Dictionary Loading
dictionary = load_tokenized_vector("commedia_tokens.json")
dict_size = len(dictionary)
print(f"Working with a dictionary size of {dict_size}")
train_data = torch.tensor(load_tokenized_vector("commedia_training_set.json"), dtype=torch.long)
val_data = torch.tensor(load_tokenized_vector("commedia_validation_set.json"), dtype=torch.long)
enc = Encoder(dictionary)

# Model generation
model = BigramLanguageModel(dict_size)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train(train_data, val_data, model, optimizer)

idx0 = torch.zeros((1,1), dtype=torch.long)
prevision = model.generate(idx0, max_new_tokens=500)
print("---- Result ----")
print(enc.decode(prevision[0].tolist()))
