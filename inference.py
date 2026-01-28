"""
Generate text from a trained MiniLLM model.

Usage:
    python inference.py                          # default: 500 chars from empty context
    python inference.py --prompt "ROMEO:"        # start from a prompt
    python inference.py --length 1000            # generate more characters
    python inference.py --temperature 0.5        # lower = more conservative
"""

import os
import argparse
import torch

# Reuse model definition from train.py
from train import MiniLLM, BLOCK_SIZE, DEVICE, VOCAB_SIZE, encode, decode


def load_model(path):
    model = MiniLLM().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def generate(model, prompt="", length=500, temperature=1.0, top_k=0):
    if prompt:
        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

    for _ in range(length):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)

    tokens = idx[0].tolist()
    return decode(tokens)


def main():
    parser = argparse.ArgumentParser(description="Generate Shakespeare with MiniLLM")
    parser.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "model.pt"), help="Path to model weights")
    parser.add_argument("--prompt", default="", help="Starting text")
    parser.add_argument("--length", type=int, default=500, help="Number of characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (lower=conservative)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0=disabled)")
    args = parser.parse_args()

    model = load_model(args.model)
    text = generate(model, args.prompt, args.length, args.temperature, args.top_k)
    print(text)


if __name__ == "__main__":
    main()
