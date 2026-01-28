import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Hyperparameters ---
# In a real scenario, these would be much larger (e.g., n_embd=4096 for Llama 2)
BATCH_SIZE = 32     # Number of sequences processed in parallel
# Read names early to determine BLOCK_SIZE
with open('names.txt', 'r') as f:
    names = f.read().strip().split('\n')
BLOCK_SIZE = max(len(n) for n in names) + 1  # max name length + 1 for END token
N_EMBD = 32         # Dimension of the embedding vector for each token
N_HEAD = 2          # Number of attention heads
HEAD_SIZE = N_EMBD // N_HEAD  # 16 per head, concatenated = 32 = N_EMBD

def apply_rope(x, head_size):
    """Apply Rotary Position Embeddings to a (B, T, head_size) tensor."""
    B, T, D = x.shape
    # Create rotation frequencies: one per pair of dimensions
    freqs = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))  # (D/2,)
    positions = torch.arange(T).float()                             # (T,)
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)            # (T, D/2)
    # Split into pairs, rotate each pair by its angle
    x = x.view(B, T, D // 2, 2)
    cos_a = torch.cos(angles).unsqueeze(0).unsqueeze(-1)  # (1, T, D/2, 1)
    sin_a = torch.sin(angles).unsqueeze(0).unsqueeze(-1)
    x0 = x[..., 0:1]  # first element of each pair
    x1 = x[..., 1:2]  # second element of each pair
    rotated = torch.cat([x0 * cos_a - x1 * sin_a,
                         x0 * sin_a + x1 * cos_a], dim=-1)
    return rotated.view(B, T, D)

class Head(nn.Module):
    """One Head of Self-Attention"""

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Apply RoPE to Q and K (not V)
        q = apply_rope(q, self.head_size)
        k = apply_rope(k, self.head_size)

        wei = q @ k.transpose(-2, -1)
        wei = wei * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel, then concatenated."""

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        # Project concatenated output back to N_EMBD
        self.proj = nn.Linear(n_heads * head_size, N_EMBD)

    def forward(self, x):
        # Run each head independently, concatenate along last dim
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, N_EMBD)
        out = self.proj(out)  # (B, T, N_EMBD)
        return out

class MLP(nn.Module):
    """Feed-forward network: expand, nonlinearity, compress."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """One transformer layer: attention + MLP, both with residual connections."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.mha = MultiHeadAttention(N_HEAD, HEAD_SIZE)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.mha(self.ln1(x))   # norm → attend → residual
        x = x + self.mlp(self.ln2(x))   # norm → MLP → residual
        return x

# --- Tokenization ---
# Character-level tokenizer with special tokens for END and PAD.
# Each name is treated as a separate sequence.

# 1. Build vocabulary: PAD=0, END=1, then a-z
PAD_TOKEN = 0
END_TOKEN = 1
chars = sorted(list(set(''.join(names))))
stoi = {ch: i + 2 for i, ch in enumerate(chars)}  # offset by 2 for PAD/END
stoi['<PAD>'] = PAD_TOKEN
stoi['<END>'] = END_TOKEN
itos = {i: ch for ch, i in stoi.items()}
VOCAB_SIZE = len(stoi)

# 3. Encode / decode functions
def encode(s):
    """Encode a name into token IDs, ending with END token."""
    return [stoi[c] for c in s] + [END_TOKEN]

def decode(tokens):
    """Decode token IDs back to string, stopping at END or PAD."""
    return ''.join([itos[t] for t in tokens if t not in (PAD_TOKEN, END_TOKEN)])

# 4. Encode all names and pad to BLOCK_SIZE
def encode_and_pad(name):
    tokens = encode(name)
    tokens += [PAD_TOKEN] * (BLOCK_SIZE - len(tokens))
    return tokens

all_data = [encode_and_pad(n) for n in names]

# 5. Split by name: 80% train, 20% test
import random
random.seed(42)
random.shuffle(all_data)
n = int(0.8 * len(all_data))
train_data = all_data[:n]
test_data = all_data[n:]

print(f"Vocabulary ({VOCAB_SIZE} tokens): PAD, END, {' '.join(chars)}")
print(f"Max name length: {BLOCK_SIZE - 1} | BLOCK_SIZE: {BLOCK_SIZE}")
print(f"Names: {len(all_data)} | Train: {len(train_data)} | Test: {len(test_data)}")
print(f"\nExample: 'emma' -> {encode('emma')} -> '{decode(encode('emma'))}'")

# --- Create training batches ---

# Embedding table: converts token IDs into learnable vectors of size N_EMBD
token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)

def get_batch(split='train'):
    """Sample a batch of whole names. Returns (input_ids, targets, mask)."""
    d = train_data if split == 'train' else test_data
    ix = torch.randint(len(d), (BATCH_SIZE,))
    batch = torch.tensor([d[i] for i in ix], dtype=torch.long)  # (B, BLOCK_SIZE)
    x = batch                                                     # input: full padded sequence
    # Target: shifted by 1. For input [e,m,m,a,END,PAD,PAD], target is [m,m,a,END,PAD,PAD,PAD]
    y = torch.cat([batch[:, 1:], torch.full((BATCH_SIZE, 1), PAD_TOKEN, dtype=torch.long)], dim=1)
    # Mask: only compute loss where target is not PAD
    mask = (y != PAD_TOKEN).float()
    return x, y, mask

# --- Model ---

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = token_embedding
        self.layers = nn.ModuleList([Block() for _ in range(4)])
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None, mask=None):
        # idx: (B, T) token IDs
        x = self.token_embedding(idx)  # (B, T, N_EMBD)
        for layer in self.layers:
            x = x + layer(x)          # Each layer + residual connection
        logits = self.lm_head(x)      # (B, T, VOCAB_SIZE)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            # Compute per-token loss, then mask out PAD positions
            loss_per_token = F.cross_entropy(logits.view(B*T, C), targets.view(B*T), reduction='none')
            loss_per_token = loss_per_token.view(B, T)
            loss = (loss_per_token * mask).sum() / mask.sum()
        return logits, loss

model = Model()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# --- Evaluation ---

@torch.no_grad()
def estimate_loss(eval_iters=50):
    """Average loss over several batches for train and test splits."""
    model.eval()
    out = {}
    for split in ['train', 'test']:
        losses = []
        for _ in range(eval_iters):
            x, y, mask = get_batch(split)
            _, loss = model(x, y, mask)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# --- Training ---

print("\nTraining...")
for step in range(5000):
    if step % 100 == 0:
        losses = estimate_loss()
        print(f"step {step:4d} | train loss: {losses['train']:.4f} | test loss: {losses['test']:.4f}")

    x, y, mask = get_batch('train')
    _, loss = model(x, y, mask)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

losses = estimate_loss()
print(f"step 5000 | train loss: {losses['train']:.4f} | test loss: {losses['test']:.4f}")