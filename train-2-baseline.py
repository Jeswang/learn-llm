import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Hyperparameters ---
BATCH_SIZE = 32
with open('names.txt', 'r') as f:
    names = f.read().strip().split('\n')
BLOCK_SIZE = max(len(n) for n in names) + 1
N_EMBD = 32
HEAD_SIZE = 16

class Head(nn.Module):
    """One Head of Self-Attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1)
        wei = wei * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out

# --- Tokenization ---
PAD_TOKEN = 0
END_TOKEN = 1
chars = sorted(list(set(''.join(names))))
stoi = {ch: i + 2 for i, ch in enumerate(chars)}
stoi['<PAD>'] = PAD_TOKEN
stoi['<END>'] = END_TOKEN
itos = {i: ch for ch, i in stoi.items()}
VOCAB_SIZE = len(stoi)

def encode(s):
    return [stoi[c] for c in s] + [END_TOKEN]

def decode(tokens):
    return ''.join([itos[t] for t in tokens if t not in (PAD_TOKEN, END_TOKEN)])

def encode_and_pad(name):
    tokens = encode(name)
    tokens += [PAD_TOKEN] * (BLOCK_SIZE - len(tokens))
    return tokens

all_data = [encode_and_pad(n) for n in names]

import random
random.seed(42)
random.shuffle(all_data)
n = int(0.8 * len(all_data))
train_data = all_data[:n]
test_data = all_data[n:]

print(f"Vocabulary ({VOCAB_SIZE} tokens): PAD, END, {' '.join(chars)}")
print(f"Max name length: {BLOCK_SIZE - 1} | BLOCK_SIZE: {BLOCK_SIZE}")
print(f"Names: {len(all_data)} | Train: {len(train_data)} | Test: {len(test_data)}")

# --- Model ---
token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = token_embedding
        self.head = Head(HEAD_SIZE)
        self.lm_head = nn.Linear(HEAD_SIZE, VOCAB_SIZE)

    def forward(self, idx, targets=None, mask=None):
        x = self.token_embedding(idx)
        x = self.head(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss_per_token = F.cross_entropy(logits.view(B*T, C), targets.view(B*T), reduction='none')
            loss_per_token = loss_per_token.view(B, T)
            loss = (loss_per_token * mask).sum() / mask.sum()
        return logits, loss

def get_batch(split='train'):
    d = train_data if split == 'train' else test_data
    ix = torch.randint(len(d), (BATCH_SIZE,))
    batch = torch.tensor([d[i] for i in ix], dtype=torch.long)
    x = batch
    y = torch.cat([batch[:, 1:], torch.full((BATCH_SIZE, 1), PAD_TOKEN, dtype=torch.long)], dim=1)
    mask = (y != PAD_TOKEN).float()
    return x, y, mask

model = Model()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

@torch.no_grad()
def estimate_loss(eval_iters=50):
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
