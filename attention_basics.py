"""
Self-Attention from scratch, explained step by step.
Run: python attention_basics.py
"""

import torch
import torch.nn.functional as F

torch.manual_seed(42)

# ============================================================
# STEP 1: The Problem — Why do we need attention?
# ============================================================
# Imagine 4 words in a sentence. Each word is just a 3-dim vector.
# The model needs a way for words to "talk to each other".
# Without attention, each word is processed in isolation.

words = torch.tensor([
    [1.0, 0.0, 0.0],   # word 0: "The"
    [0.0, 1.0, 0.0],   # word 1: "cat"
    [0.0, 0.0, 1.0],   # word 2: "sat"
    [1.0, 1.0, 0.0],   # word 3: "down"
])
T, C = words.shape  # T=4 tokens, C=3 channels
print("=== STEP 1: Raw word vectors ===")
print(words)
print()

# ============================================================
# STEP 2: The simplest "attention" — just average past words
# ============================================================
# For each word, average it with all *previous* words (including itself).
# Word 0 sees only itself. Word 1 sees words 0-1. Word 2 sees 0-2. etc.
# This is called a "bag of words" approach — no learning involved.

print("=== STEP 2: Simple averaging (no learning) ===")
avg_out = torch.zeros(T, C)
for i in range(T):
    avg_out[i] = words[:i + 1].mean(dim=0)
    print(f"  word {i}: avg of words 0..{i} = {avg_out[i].tolist()}")
print()

# ============================================================
# STEP 3: Do the same thing with matrix multiplication
# ============================================================
# Instead of a loop, we use a lower-triangular matrix.
# Each row tells us: what fraction of each word to include.

tril = torch.tril(torch.ones(T, T))
weights = tril / tril.sum(dim=1, keepdim=True)  # normalize rows to sum to 1

print("=== STEP 3: Weight matrix (who talks to whom) ===")
print(weights)
print()
print("Row 0: word 0 sees [100% of word 0]")
print("Row 1: word 1 sees [50% of word 0, 50% of word 1]")
print("Row 2: word 2 sees [33% each of words 0, 1, 2]")
print()

mat_out = weights @ words  # (4,4) @ (4,3) -> (4,3)
print("Result (same as step 2):")
print(mat_out)
print()

# ============================================================
# STEP 4: Softmax — a better way to build the weight matrix
# ============================================================
# Instead of manual division, use softmax.
# Start with zeros, mask future positions to -inf, then softmax.

print("=== STEP 4: Using softmax + masking ===")
raw = torch.zeros(T, T)
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()  # upper triangle = future
raw = raw.masked_fill(mask, float('-inf'))
print("Before softmax (future = -inf):")
print(raw)
print()

weights2 = F.softmax(raw, dim=-1)
print("After softmax (future becomes 0, rest = uniform):")
print(weights2)
print()

# ============================================================
# STEP 5: Learned attention — Query, Key, Value
# ============================================================
# The big idea: instead of uniform averaging, let words CHOOSE
# what to pay attention to. We do this with three projections:
#
#   Query = "what am I looking for?"
#   Key   = "what do I contain?"
#   Value = "what do I give if someone attends to me?"
#
# Attention score = how well Query matches Key.

print("=== STEP 5: Query-Key-Value attention ===")

head_size = 4  # small for demonstration

# These are the learnable weight matrices (normally trained via backprop)
torch.manual_seed(42)
W_q = torch.randn(C, head_size)   # (3, 4)
W_k = torch.randn(C, head_size)   # (3, 4)
W_v = torch.randn(C, head_size)   # (3, 4)

Q = words @ W_q   # (4, 4) — each word's query
K = words @ W_k   # (4, 4) — each word's key
V = words @ W_v   # (4, 4) — each word's value

print(f"Q (what each word looks for):\n{Q}\n")
print(f"K (what each word advertises):\n{K}\n")

# Attention scores: dot product of Q and K
scores = Q @ K.T   # (4, 4) — score[i][j] = how much word i attends to word j
scores = scores / head_size**0.5  # scale to prevent large values

print(f"Raw attention scores (Q @ K^T / sqrt({head_size})):")
print(scores)
print()

# Mask future positions
scores = scores.masked_fill(mask, float('-inf'))

# Softmax to get attention weights
attn = F.softmax(scores, dim=-1)
print("Attention weights (after mask + softmax):")
print(attn)
print()
print("Notice: each row sums to 1, and future positions are 0.")
print(f"Row sums: {attn.sum(dim=-1).tolist()}")
print()

# Final output: weighted combination of Values
output = attn @ V  # (4, 4) — each word is now a blend of relevant values
print(f"Output (attention-weighted values):\n{output}")
print()

# ============================================================
# STEP 6: Why this matters
# ============================================================
print("=== STEP 6: Summary ===")
print("""
Without attention:  Each word is processed alone.
With averaging:     Each word sees equal blend of past — no selectivity.
With Q/K/V:         Each word CHOOSES what's relevant from the past.

During training, the model learns W_q, W_k, W_v so that:
  - Verbs attend to their subjects
  - Adjectives attend to their nouns
  - Closing quotes attend to opening quotes
  - etc.

This is ONE attention head. A full transformer uses:
  - Multiple heads (each learning different patterns)
  - Feed-forward layers (to process gathered information)
  - Residual connections (to preserve original signal)
  - Layer norms (to stabilize training)
  - Stacked N times (our model uses 6 layers)
""")
