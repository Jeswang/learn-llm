# Building a Transformer Step by Step

Incrementally building a character-level transformer for name generation using [karpathy/makemore](https://github.com/karpathy/makemore) names dataset (32,033 names). Each step adds one architectural improvement. All models trained for 5,000 steps with batch size 32, AdamW optimizer (lr=1e-3), and per-name padding with masked loss.

## Results

| File | Config | N_EMBD | HEAD_SIZE | Heads | Layers | Params | Train | Test |
|---|---|---|---|---|---|---|---|---|
| `train-2-baseline.py` | baseline | 32 | 16 | 1 | 1 | 2,908 | 2.35 | 2.35 |
| `train-2-double-embd.py` | double embd | 64 | 32 | 1 | 1 | 8,860 | 2.34 | 2.34 |
| `train-2-2head.py` | 2 heads | 32 | 16 | 2 | 1 | 5,948 | 2.25 | 2.23 |
| `train-2-2head-4layer.py` | 4 layers | 32 | 16 | 2 | 4 | 18,332 | 2.00 | 2.04 |
| `train-2-2head-4layer-mlp.py` | + MLP | 32 | 16 | 2 | 4 | 51,740 | 1.97 | 2.02 |
| `train-2-2head-4layer-mlp-ln.py` | + LayerNorm | 32 | 16 | 2 | 4 | 52,252 | 1.96 | 1.99 |
| `train-2-2head-4layer-mlp-ln-rope.py` | + RoPE | 32 | 16 | 2 | 4 | 52,252 | 1.94 | 1.98 |
| `train-2.py` | + GELU | 32 | 16 | 2 | 4 | 52,252 | 1.94 | 1.94 |

Makemore's default transformer achieves ~1.92 test loss with N_EMBD=64, 4 heads, 4 layers.

## Key Findings

### Depth beats width

Doubling embedding size from 32 to 64 (3x params) gave almost no improvement (2.35 -> 2.34). Adding a second attention head with fewer total params (5,948 vs 8,860) dropped loss by 0.12. Stacking 4 layers was the single biggest improvement, dropping test loss from 2.23 to 2.04. The model benefits far more from multiple layers of processing than from wider representations at a single layer.

### Data handling matters most

Before adding per-name padding, our best model achieved 2.36 test loss. After switching to per-name padding with masked loss (same architecture), it dropped to 1.94. This was a larger improvement than all architectural changes combined. The reason: without padding, the model wasted capacity trying to predict across name boundaries -- an impossible task that added noise to every gradient update.

### MLP adds capacity but needs regularization

Adding the feed-forward network (MLP) to each layer tripled the parameter count (18k -> 52k) but only modestly improved results. It also widened the train-test gap (2.00/2.04 -> 1.97/2.02), suggesting mild overfitting. The MLP lets the model transform representations nonlinearly after attention gathers information, but at this small scale the effect is limited.

### LayerNorm and RoPE help incrementally

LayerNorm stabilized training and closed the train-test gap slightly. RoPE (Rotary Position Embeddings) gave the model awareness of character positions without adding any parameters. Neither was dramatic at this scale, but both are essential for larger models -- LayerNorm enables training deep networks, and RoPE enables generalization to longer sequences.

### GELU vs ReLU is negligible at small scale

Switching from ReLU to GELU activation in the MLP had no measurable effect. The smoother gradient flow matters more when networks are deeper and wider.

## Architecture Summary

The final model (`train-2.py`) is a proper transformer decoder:

```
Input tokens
    -> Token Embedding (28 vocab -> 32 dim)
    -> 4x Transformer Blocks:
        -> LayerNorm -> Multi-Head Attention (2 heads, RoPE) -> Residual
        -> LayerNorm -> MLP (32 -> 128 -> 32, GELU) -> Residual
    -> Linear (32 -> 28 vocab)
    -> Cross-entropy loss (masked on PAD tokens)
```

## What the loss means

A loss of 1.94 means the model assigns ~14.3% probability on average to the correct next character (`e^(-1.94)`). Random guessing over 27 characters would give ~3.7% (loss = 3.30). Perfect prediction is impossible because many positions are genuinely ambiguous -- after "ma", the next character could be r, d, k, x, t, and many others.
