"""
Export Image Captioning Model to ONNX format for browser inference
Using legacy exporter for better browser compatibility
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

os.environ["PYTORCH_ONNX_USE_OLD_EXPORTER"] = "1"

# --- Hyperparameters (must match training) ---
IMAGE_SIZE = 128
PATCH_SIZE = 8
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
MAX_CAPTION_LEN = 48

# --- Model Definition ---
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, n_embd):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, n_embd)
        self.pos_embd = nn.Parameter(torch.randn(1, self.n_patches, n_embd) * 0.02)
        self.image_size = image_size

    def forward(self, x):
        B = x.shape[0]
        p = self.patch_size
        # Manual patch extraction (more ONNX-friendly)
        patches = []
        for i in range(0, self.image_size, p):
            for j in range(0, self.image_size, p):
                patch = x[:, :, i:i+p, j:j+p]  # (B, 3, p, p)
                patch = patch.reshape(B, -1)  # (B, 3*p*p)
                patches.append(patch)
        x = torch.stack(patches, dim=1)  # (B, n_patches, 3*p*p)
        x = self.proj(x) + self.pos_embd
        return x


class Head(nn.Module):
    def __init__(self, head_size, n_embd, is_causal=False, block_size=None):
        super().__init__()
        self.is_causal = is_causal
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        if is_causal and block_size:
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.block_size = block_size

    def forward(self, x, context=None):
        B, T, C = x.shape
        q = self.query(x)
        if context is not None:
            k = self.key(context)
            v = self.value(context)
        else:
            k = self.key(x)
            v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
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
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


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


class ImageEncoder(nn.Module):
    def __init__(self, image_size, patch_size, n_embd, n_head, n_layer):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, n_embd)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(n_embd, n_head) for _ in range(n_layer // 2)
        ])

    def forward(self, img):
        x = self.patch_embed(img)
        for block in self.encoder_blocks:
            x = block(x)
        return x


class TextDecoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_caption_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(max_caption_len, n_embd)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(n_embd, n_head, max_caption_len) for _ in range(n_layer // 2)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.max_len = max_caption_len

    def forward(self, tokens, encoder_out):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.token_embed(tokens) + self.pos_embed(pos)
        for block in self.decoder_blocks:
            x = block(x, encoder_out)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# Original model for loading weights
class OriginalPatchEmbedding(nn.Module):
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


class OriginalMLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class OriginalEncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.mha = MultiHeadAttention(n_head, head_size, n_embd, is_causal=False)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = OriginalMLP(n_embd)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class OriginalDecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.self_attn = MultiHeadAttention(n_head, head_size, n_embd, is_causal=True, block_size=block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.cross_attn = MultiHeadAttention(n_head, head_size, n_embd, is_causal=False)
        self.ln3 = nn.LayerNorm(n_embd)
        self.mlp = OriginalMLP(n_embd)

    def forward(self, x, encoder_out):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), encoder_out)
        x = x + self.mlp(self.ln3(x))
        return x


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, image_size, patch_size, n_embd, n_head, n_layer, max_caption_len):
        super().__init__()
        self.max_caption_len = max_caption_len
        self.patch_embed = OriginalPatchEmbedding(image_size, patch_size, n_embd)
        self.encoder_blocks = nn.ModuleList([
            OriginalEncoderBlock(n_embd, n_head) for _ in range(n_layer // 2)
        ])
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(max_caption_len, n_embd)
        self.decoder_blocks = nn.ModuleList([
            OriginalDecoderBlock(n_embd, n_head, max_caption_len) for _ in range(n_layer // 2)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


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

    # Save vocabulary
    vocab_data = {
        'stoi': stoi,
        'itos': {str(k): v for k, v in itos.items()},
        'vocab_size': vocab_size
    }
    with open('vocab.json', 'w') as f:
        json.dump(vocab_data, f)
    print("Saved vocabulary to vocab.json")

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

    # Create new encoder/decoder with ONNX-friendly operations
    encoder = ImageEncoder(IMAGE_SIZE, PATCH_SIZE, N_EMBD, N_HEAD, N_LAYER)
    decoder = TextDecoder(vocab_size, N_EMBD, N_HEAD, N_LAYER, MAX_CAPTION_LEN)

    # Copy weights - patch embedding
    encoder.patch_embed.proj.load_state_dict(model.patch_embed.proj.state_dict())
    encoder.patch_embed.pos_embd.data = model.patch_embed.pos_embd.data.clone()

    # Encoder blocks
    for i in range(len(encoder.encoder_blocks)):
        enc_block = encoder.encoder_blocks[i]
        orig_block = model.encoder_blocks[i]
        enc_block.ln1.load_state_dict(orig_block.ln1.state_dict())
        enc_block.mha.load_state_dict(orig_block.mha.state_dict())
        enc_block.ln2.load_state_dict(orig_block.ln2.state_dict())
        enc_block.mlp.fc1.weight.data = orig_block.mlp.net[0].weight.data.clone()
        enc_block.mlp.fc1.bias.data = orig_block.mlp.net[0].bias.data.clone()
        enc_block.mlp.fc2.weight.data = orig_block.mlp.net[2].weight.data.clone()
        enc_block.mlp.fc2.bias.data = orig_block.mlp.net[2].bias.data.clone()

    # Decoder
    decoder.token_embed.load_state_dict(model.token_embed.state_dict())
    decoder.pos_embed.load_state_dict(model.pos_embed.state_dict())
    decoder.ln_f.load_state_dict(model.ln_f.state_dict())
    decoder.lm_head.load_state_dict(model.lm_head.state_dict())

    for i in range(len(decoder.decoder_blocks)):
        dec_block = decoder.decoder_blocks[i]
        orig_block = model.decoder_blocks[i]
        dec_block.ln1.load_state_dict(orig_block.ln1.state_dict())
        dec_block.self_attn.load_state_dict(orig_block.self_attn.state_dict())
        dec_block.ln2.load_state_dict(orig_block.ln2.state_dict())
        dec_block.cross_attn.load_state_dict(orig_block.cross_attn.state_dict())
        dec_block.ln3.load_state_dict(orig_block.ln3.state_dict())
        dec_block.mlp.fc1.weight.data = orig_block.mlp.net[0].weight.data.clone()
        dec_block.mlp.fc1.bias.data = orig_block.mlp.net[0].bias.data.clone()
        dec_block.mlp.fc2.weight.data = orig_block.mlp.net[2].weight.data.clone()
        dec_block.mlp.fc2.bias.data = orig_block.mlp.net[2].bias.data.clone()

    encoder.eval()
    decoder.eval()

    # Test before export
    print("Testing models before export...")
    test_img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        features = encoder(test_img)
        print(f"Encoder output: {features.shape}")
        test_tokens = torch.tensor([[1]])  # START token
        logits = decoder(test_tokens, features)
        print(f"Decoder output: {logits.shape}")

    # Export with legacy exporter
    print("\nExporting encoder to ONNX (opset 12)...")
    dummy_img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        encoder,
        dummy_img,
        "encoder.onnx",
        input_names=['image'],
        output_names=['features'],
        opset_version=12,
        do_constant_folding=True
    )
    print("Saved encoder.onnx")

    print("Exporting decoder to ONNX (opset 12)...")
    dummy_tokens = torch.tensor([[1]], dtype=torch.long)
    dummy_features = torch.randn(1, 256, N_EMBD)
    torch.onnx.export(
        decoder,
        (dummy_tokens, dummy_features),
        "decoder.onnx",
        input_names=['tokens', 'encoder_features'],
        output_names=['logits'],
        opset_version=12,
        do_constant_folding=True
    )
    print("Saved decoder.onnx")

    # Check file sizes
    import os
    enc_size = os.path.getsize("encoder.onnx") / 1024
    dec_size = os.path.getsize("decoder.onnx") / 1024
    print(f"\nEncoder size: {enc_size:.1f} KB")
    print(f"Decoder size: {dec_size:.1f} KB")
    print(f"Total: {enc_size + dec_size:.1f} KB")

    print("\nExport complete!")


if __name__ == "__main__":
    main()
