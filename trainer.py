import torch

class Trainer:
    def __init__(self, model, optimizer, train_data, val_data, eval_interval = 100, estimate_iters=1000):
        self.model = model
        self.optimizer = optimizer
        self.eval_interval = eval_interval
        self.estimate_iters = estimate_iters
        self.train_data = train_data
        self.val_data = val_data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __call__(self, train_iters, batch_size, block_size):
        self.train(train_iters, batch_size, block_size)
    
    def train(self, train_iters, batch_size, block_size):
        self.eval_iters = train_iters
        for step in range(train_iters):
            self.on_step(step, batch_size, block_size)
            self.train_step(batch_size, block_size)

    def train_step(self, batch_size, block_size):
        xb, yb = self.get_batch(self.train_data, batch_size, block_size)
        _, loss = self.model(xb, yb)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def estimate_loss(self, batch_size, block_size):
        out = {}
        self.model.eval()
        for type, split in [("train", self.train_data), ("val", self.val_data)]:
            losses = torch.zeros(self.estimate_iters)
            for k in range(self.estimate_iters):
                xb, yb = self.get_batch(split, batch_size, block_size)
                logits, loss = self.model(xb, yb)
                losses[k] = loss.item()
            out[type] = losses.mean()
        self.model.train()
        return out

    def get_batch(self, data, batch_size, block_size):
        split_pos = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack( [data[pos : pos+block_size] for pos in split_pos])
        y = torch.stack( [data[pos+1 : pos+block_size+1] for pos in split_pos])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    def on_step(self, step, batch_size, block_size):
        if step % self.eval_interval == 0:
            losses = self.estimate_loss(batch_size, block_size)
            print(f"Step {step}: train loss={losses['train']}; validation loss={losses['val']}", 
                flush = True)
