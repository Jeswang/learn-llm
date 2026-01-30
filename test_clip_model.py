"""
Test CLIP-based model on specific images
"""
import torch
import clip
from PIL import Image
import json

# Load model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load CLIP
print("Loading CLIP...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Load vocabulary
with open('vocab_clip.json', 'r') as f:
    vocab_data = json.load(f)
stoi = vocab_data['stoi']
itos = {int(k): v for k, v in vocab_data['itos'].items()}

PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2

def decode(tokens):
    chars = []
    for t in tokens:
        if t == END_TOKEN:
            break
        if t not in (PAD_TOKEN, START_TOKEN):
            chars.append(itos.get(t, '?'))
    return ''.join(chars)

# Rebuild model architecture
import torch.nn as nn
import torch.nn.functional as F

N_EMBD = 256
N_HEAD = 8
N_LAYER = 6
MAX_CAPTION_LEN = 48
vocab_size = len(stoi)

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

class CLIPCaptioningModel(nn.Module):
    def __init__(self, vocab_size, clip_model, n_embd, n_head, n_layer, max_caption_len, freeze_encoder=True):
        super().__init__()
        self.max_caption_len = max_caption_len
        self.clip_model = clip_model
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        self.visual_proj = nn.Linear(768, n_embd)
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(max_caption_len, n_embd)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(n_embd, n_head, max_caption_len) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def encode_image(self, img):
        with torch.set_grad_enabled(not self.freeze_encoder):
            visual = self.clip_model.visual
            x = visual.conv1(img.type(visual.conv1.weight.dtype))
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls_token, x], dim=1)
            x = x + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)
        x = self.visual_proj(x.float())
        return x

    @torch.no_grad()
    def generate(self, img, start_token, end_token, max_len=32):
        self.eval()
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img_features = self.encode_image(img)
        tokens = [start_token]
        for _ in range(max_len - 1):
            x = torch.tensor([tokens], device=img.device)
            pos = torch.arange(len(tokens), device=img.device)
            x = self.token_embed(x) + self.pos_embed(pos)
            for block in self.decoder_blocks:
                x = block(x, img_features)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            next_token = logits[0, -1].argmax().item()
            tokens.append(next_token)
            if next_token == end_token:
                break
        return tokens

# Create and load model
print("Creating model...")
model = CLIPCaptioningModel(
    vocab_size=vocab_size,
    clip_model=clip_model,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    max_caption_len=MAX_CAPTION_LEN,
    freeze_encoder=True
).to(device)

print("Loading weights...")
checkpoint = torch.load('image_caption_clip_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test on Flickr8k validation images
from datasets import load_dataset
print("\nLoading Flickr8k validation set...")
ds = load_dataset('jxie/flickr8k', split='validation')

# Find complex scenes (the "failure cases")
print("\n" + "="*60)
print("Testing CLIP model on various images")
print("="*60)

# Test first few validation images
for i in range(10):
    img = ds[i]['image'].convert('RGB')
    img_tensor = clip_preprocess(img).to(device)

    tokens = model.generate(img_tensor, START_TOKEN, END_TOKEN)
    generated = decode(tokens)

    actual = ds[i]['caption_0']

    print(f"\nImage {i}:")
    print(f"  Generated: '{generated}'")
    print(f"  Actual:    '{actual}'")
