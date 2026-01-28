"""
Minimal character-level language model trained on Shakespeare.
Only dependency: PyTorch.
"""

import os
import math
import time
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------- Hyperparameters ---------------
BATCH_SIZE = 32
BLOCK_SIZE = 128        # context length
N_EMBD = 192
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.1
LR = 3e-4
MAX_ITERS = 3000
EVAL_INTERVAL = 500
EVAL_ITERS = 100
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "input.txt")

# --------------- Data ---------------

def get_data():
    if not os.path.exists(DATA_PATH):
        print("Downloading tiny Shakespeare...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r") as f:
        return f.read()

text = get_data()
chars = sorted(set(text))
VOCAB_SIZE = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# --------------- Model ---------------

class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        w = q @ k.transpose(-2, -1) * (C ** -0.5)
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = self.dropout(F.softmax(w, dim=-1))
        return w @ v


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = N_EMBD // N_HEAD
        self.heads = nn.ModuleList([SelfAttention(head_size) for _ in range(N_HEAD)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = MultiHeadAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.ff = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB_SIZE)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=DEVICE))
        x = self.ln_f(self.blocks(tok + pos))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# --------------- Training ---------------

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def progress_bar(current, total, width=30):
    frac = current / total
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    return f"|{bar}| {current}/{total} ({frac:.0%})"


def main():
    print(f"Device: {DEVICE}", flush=True)
    model = MiniLLM().to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count / 1e6:.2f}M", flush=True)
    print(flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    t_start = time.time()

    for step in range(MAX_ITERS):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            elapsed = time.time() - t_start
            if step > 0:
                eta = elapsed / step * (MAX_ITERS - step)
                timing = f"elapsed {fmt_time(elapsed)} | ETA {fmt_time(eta)}"
            else:
                timing = "elapsed 0s"
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {timing}")
        elif step % 100 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / step * (MAX_ITERS - step) if step > 0 else 0
            print(f"  {progress_bar(step, MAX_ITERS)} loss {loss.item():.4f} | ETA {fmt_time(eta)}", flush=True)

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Final eval
    total_time = time.time() - t_start
    losses = estimate_loss(model)
    print(f"step {MAX_ITERS:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | total {fmt_time(total_time)}")

    # Generate sample
    print("\n--- Generated Shakespeare ---\n")
    ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(model.generate(ctx, max_new_tokens=500)[0].tolist()))

    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "model.pt"))
    print("\nModel saved to model.pt")


if __name__ == "__main__":
    main()
