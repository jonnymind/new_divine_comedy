import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, voc_size, block_size, n_embed, n_heads, n_layers, dropout=0.2):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.voc_size = voc_size
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.block_size = block_size
        self.n_layers = n_layers
        # In our case, the embeddings are the probability distribution of each next-token (column)
        # given the current token (row). The sum-total the value in each row must be 1.
        # Call it next-token probability distribution (NTPD) matrix

        # could be (voc_size, voc_size), but in practice we can ignore the less commonly following tokens.
        self.token_embedding_table = nn.Embedding(voc_size, n_embed, device = self.device)
        self.position_embedding_table = nn.Embedding(block_size, n_embed, device = self.device)
        # ... but to do that we need a index-map so that the numbers in each row xi don't represent the
        # probability of the i-th token to appear after y-i, but an index in an header table.
        self.blocks = nn.Sequential( *[Block(n_embed, n_heads, dropout) for _ in range(n_layers)],
            nn.LayerNorm(n_embed)
        )
        self.sa_head = nn.Linear(n_embed, voc_size, device = self.device)
        self.to(self.device)

    def forward(self, inputs, targets = None):
        Btc, Blk = inputs.shape
        inputs = inputs.to(self.device)
        # Input and targets are both (batch_size, block_size)
        # This gives us a matrix with every token in every batch replaced by the respective NTPD
        tok_emb = self.token_embedding_table(inputs).to(self.device)  # (Bch, Blk, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(Blk, device=self.device)) # (Blk, n_embed)
        x = tok_emb + pos_emb
        blocks_out = self.blocks(x) # (Bch, Blk, n_embeds)
        logits = self.sa_head(blocks_out) # (Bch, Blk, Voc)

        if targets == None:
            loss = None
        else:
            targets = targets.to(self.device)
            Bch, Blk, Voc = logits.shape
            # Cross-entropy wants a vector or a 2D matrix, we need to flatten the batch dimension.
            entropy_logits = logits.view(Bch * Blk, Voc)
            targets = targets.view(Bch * Blk)
            # Cross-entropy is a good and fast measure of the difference between two prob-dists.
            loss = F.cross_entropy(entropy_logits, targets)
        return logits, loss
    
    def generate(self, inputs, max_new_tokens):
        # Inputs should be in (Btc, Blk)
        inputs = inputs.to(self.device)
        for _ in range(max_new_tokens):
            idx_context = inputs[:, -self.block_size: ]
            logits, loss = self(idx_context) 
            # Get the NTPD for the last token in each block.
            last_logits = logits[:, -1, :]  # (Btc, Voc)
            # Turn their predictions in a distribution 
            probs = F.softmax(last_logits, dim=-1)  # (Btc, Voc)
            # Extracts one token index according with the probability of happening next
            idx_next = torch.multinomial(probs, num_samples=1).to(self.device)  # (Btc)
            # Add the extracted tokens to the existing dataset
            inputs = torch.cat((inputs, idx_next), dim=1) # (Btc)
        return inputs


class Block(nn.Module):
    """Trasformer block: communication followed by computation"""
    def __init__(self, n_embed, n_heads, dropout=0.2):
        super().__init__()
        self.sa = MultiHeadAttention(n_embed, n_heads, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 


class FeedForward(nn.Module):
    """Moves the token weights one step forward"""
    def __init__(self, n_embed, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, num_heads, dropout = 0.2):
        super().__init__()
        head_size = n_embed // num_heads
        self.heads = nn.ModuleList([SelfAttHead(n_embed, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    

class SelfAttHead(nn.Module):
    """ one head of self attention """
    
    def __init__(self, n_embed, head_size, dropout = 0.2):
        super().__init__()
        
        self.n_embed = n_embed
        self.key = nn.Linear(self.n_embed, head_size, bias = False)
        self.query = nn.Linear(self.n_embed, head_size, bias = False)
        self.value = nn.Linear(self.n_embed, head_size, bias = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        if not hasattr(self, 'tril'):
            self.register_buffer('tril', torch.tril(torch.ones(T, T)))
            if torch.cuda.is_available():
                self.tril = self.tril.to("cuda")

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
    




