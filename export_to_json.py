"""
Export Image Captioning Model weights to JSON for direct JavaScript loading
This avoids ONNX compatibility issues entirely
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np

# --- Hyperparameters (must match training) ---
IMAGE_SIZE = 128
PATCH_SIZE = 8
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
MAX_CAPTION_LEN = 48

# --- Model Definition (same as training) ---
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, n_embd):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, n_embd)
        self.pos_embd = nn.Parameter(torch.randn(1, self.n_patches, n_embd) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(B, -1, C * p * p)
        x = self.proj(x) + self.pos_embd
        return x


class Head(nn.Module):
    def __init__(self, head_size, n_embd, is_causal=False, block_size=None):
        super().__init__()
        self.is_causal = is_causal
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        if is_causal and block_size:
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, context=None):
        B, T, C = x.shape
        q = self.query(x)
        if context is not None:
            k = self.key(context)
            v = self.value(context)
        else:
            k = self.key(x)
            v = self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        if self.is_causal:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, n_embd, is_causal=False, block_size=None):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, is_causal, block_size) for _ in range(n_heads)
        ])
        self.proj = nn.Linear(n_heads * head_size, n_embd)

    def forward(self, x, context=None):
        out = torch.cat([h(x, context) for h in self.heads], dim=-1)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.mha = MultiHeadAttention(n_head, head_size, n_embd, is_causal=False)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.self_attn = MultiHeadAttention(n_head, head_size, n_embd, is_causal=True, block_size=block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.cross_attn = MultiHeadAttention(n_head, head_size, n_embd, is_causal=False)
        self.ln3 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x, encoder_out):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), encoder_out)
        x = x + self.mlp(self.ln3(x))
        return x


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, image_size, patch_size, n_embd, n_head, n_layer, max_caption_len):
        super().__init__()
        self.max_caption_len = max_caption_len
        self.patch_embed = PatchEmbedding(image_size, patch_size, n_embd)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(n_embd, n_head) for _ in range(n_layer // 2)
        ])
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(max_caption_len, n_embd)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(n_embd, n_head, max_caption_len) for _ in range(n_layer // 2)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


def tensor_to_list(t):
    """Convert tensor to nested list for JSON"""
    return t.detach().cpu().numpy().tolist()


def main():
    # Build vocabulary
    from datasets import load_dataset
    print("Loading dataset to build vocabulary...")
    ds = load_dataset('jxie/flickr8k', split='train')

    all_chars = set()
    for item in ds:
        for i in range(5):
            caption = item[f'caption_{i}']
            all_chars.update(caption.lower())

    chars = sorted(list(all_chars))
    stoi = {ch: i + 3 for i, ch in enumerate(chars)}
    stoi['<PAD>'] = 0
    stoi['<START>'] = 1
    stoi['<END>'] = 2
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(stoi)
    print(f"Vocabulary size: {vocab_size}")

    # Load trained model
    print("Loading trained model...")
    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        max_caption_len=MAX_CAPTION_LEN
    )
    model.load_state_dict(torch.load('image_caption_model.pth', map_location='cpu'))
    model.eval()
    print("Model loaded!")

    # Export weights to JSON structure
    print("Exporting weights...")

    weights = {
        'config': {
            'image_size': IMAGE_SIZE,
            'patch_size': PATCH_SIZE,
            'n_embd': N_EMBD,
            'n_head': N_HEAD,
            'n_layer': N_LAYER,
            'max_caption_len': MAX_CAPTION_LEN,
            'vocab_size': vocab_size,
            'n_patches': (IMAGE_SIZE // PATCH_SIZE) ** 2,
            'head_size': N_EMBD // N_HEAD,
        },
        'vocab': {
            'stoi': stoi,
            'itos': {str(k): v for k, v in itos.items()},
        },
        'encoder': {
            'patch_embed': {
                'proj_weight': tensor_to_list(model.patch_embed.proj.weight),
                'proj_bias': tensor_to_list(model.patch_embed.proj.bias),
                'pos_embd': tensor_to_list(model.patch_embed.pos_embd.squeeze(0)),
            },
            'blocks': []
        },
        'decoder': {
            'token_embed': tensor_to_list(model.token_embed.weight),
            'pos_embed': tensor_to_list(model.pos_embed.weight),
            'blocks': [],
            'ln_f_weight': tensor_to_list(model.ln_f.weight),
            'ln_f_bias': tensor_to_list(model.ln_f.bias),
            'lm_head_weight': tensor_to_list(model.lm_head.weight),
            'lm_head_bias': tensor_to_list(model.lm_head.bias),
        }
    }

    # Export encoder blocks
    for i, block in enumerate(model.encoder_blocks):
        block_data = {
            'ln1_weight': tensor_to_list(block.ln1.weight),
            'ln1_bias': tensor_to_list(block.ln1.bias),
            'ln2_weight': tensor_to_list(block.ln2.weight),
            'ln2_bias': tensor_to_list(block.ln2.bias),
            'mha_heads': [],
            'mha_proj_weight': tensor_to_list(block.mha.proj.weight),
            'mha_proj_bias': tensor_to_list(block.mha.proj.bias),
            'mlp_fc1_weight': tensor_to_list(block.mlp.net[0].weight),
            'mlp_fc1_bias': tensor_to_list(block.mlp.net[0].bias),
            'mlp_fc2_weight': tensor_to_list(block.mlp.net[2].weight),
            'mlp_fc2_bias': tensor_to_list(block.mlp.net[2].bias),
        }
        for head in block.mha.heads:
            block_data['mha_heads'].append({
                'q_weight': tensor_to_list(head.query.weight),
                'k_weight': tensor_to_list(head.key.weight),
                'v_weight': tensor_to_list(head.value.weight),
            })
        weights['encoder']['blocks'].append(block_data)

    # Export decoder blocks
    for i, block in enumerate(model.decoder_blocks):
        block_data = {
            'ln1_weight': tensor_to_list(block.ln1.weight),
            'ln1_bias': tensor_to_list(block.ln1.bias),
            'ln2_weight': tensor_to_list(block.ln2.weight),
            'ln2_bias': tensor_to_list(block.ln2.bias),
            'ln3_weight': tensor_to_list(block.ln3.weight),
            'ln3_bias': tensor_to_list(block.ln3.bias),
            'self_attn_heads': [],
            'self_attn_proj_weight': tensor_to_list(block.self_attn.proj.weight),
            'self_attn_proj_bias': tensor_to_list(block.self_attn.proj.bias),
            'cross_attn_heads': [],
            'cross_attn_proj_weight': tensor_to_list(block.cross_attn.proj.weight),
            'cross_attn_proj_bias': tensor_to_list(block.cross_attn.proj.bias),
            'mlp_fc1_weight': tensor_to_list(block.mlp.net[0].weight),
            'mlp_fc1_bias': tensor_to_list(block.mlp.net[0].bias),
            'mlp_fc2_weight': tensor_to_list(block.mlp.net[2].weight),
            'mlp_fc2_bias': tensor_to_list(block.mlp.net[2].bias),
        }
        for head in block.self_attn.heads:
            block_data['self_attn_heads'].append({
                'q_weight': tensor_to_list(head.query.weight),
                'k_weight': tensor_to_list(head.key.weight),
                'v_weight': tensor_to_list(head.value.weight),
            })
        for head in block.cross_attn.heads:
            block_data['cross_attn_heads'].append({
                'q_weight': tensor_to_list(head.query.weight),
                'k_weight': tensor_to_list(head.key.weight),
                'v_weight': tensor_to_list(head.value.weight),
            })
        weights['decoder']['blocks'].append(block_data)

    # Save as JSON
    print("Saving model.json...")
    with open('model.json', 'w') as f:
        json.dump(weights, f)

    import os
    size_mb = os.path.getsize('model.json') / (1024 * 1024)
    print(f"model.json size: {size_mb:.2f} MB")

    # Also save compressed binary format
    print("Saving model.bin (binary)...")
    state_dict = model.state_dict()
    all_weights = []
    weight_info = []
    offset = 0

    for name, param in state_dict.items():
        arr = param.numpy().astype(np.float32)
        shape = list(arr.shape)
        arr_flat = arr.flatten()
        weight_info.append({
            'name': name,
            'shape': shape,
            'offset': offset,
            'length': len(arr_flat)
        })
        all_weights.append(arr_flat)
        offset += len(arr_flat)

    all_weights = np.concatenate(all_weights)
    all_weights.tofile('model.bin')

    # Save weight info
    with open('model_info.json', 'w') as f:
        json.dump({
            'config': weights['config'],
            'vocab': weights['vocab'],
            'weights': weight_info
        }, f)

    bin_size = os.path.getsize('model.bin') / (1024 * 1024)
    info_size = os.path.getsize('model_info.json') / 1024
    print(f"model.bin size: {bin_size:.2f} MB")
    print(f"model_info.json size: {info_size:.1f} KB")
    print(f"Total binary: {bin_size + info_size/1024:.2f} MB")

    print("\nExport complete!")
    print("Files: model.json (full), model.bin + model_info.json (binary)")


if __name__ == "__main__":
    main()
