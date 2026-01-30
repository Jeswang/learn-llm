"""
Minimal Image Captioning Transformer
- Vision Encoder: Simple patch embedding + transformer
- Text Decoder: Autoregressive transformer (similar to train-2.py)
- Cross-attention between image and text

Uses Flickr8k dataset (~8000 images, 5 captions each)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# --- Device ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Hyperparameters ---
BATCH_SIZE = 16
IMAGE_SIZE = 128         # Larger for better quality
PATCH_SIZE = 8           # 8x8 patches -> 64 patches per image
N_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 64
N_EMBD = 128
N_HEAD = 4
HEAD_SIZE = N_EMBD // N_HEAD
N_LAYER = 4
MAX_CAPTION_LEN = 48
LEARNING_RATE = 3e-4
NUM_EPOCHS = 30

# --- Tokenization (character-level like train-2.py) ---
PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2

# --- Dataset ---
class Flickr8kDataset(Dataset):
    def __init__(self, split='train', image_size=64):
        print(f"Loading Flickr8k {split} split...")
        self.ds = load_dataset('jxie/flickr8k', split=split)
        self.image_size = image_size

        # Build vocabulary from all captions
        print("Building vocabulary...")
        all_chars = set()
        for item in self.ds:
            for i in range(5):
                caption = item[f'caption_{i}']
                all_chars.update(caption.lower())

        self.chars = sorted(list(all_chars))
        self.stoi = {ch: i + 3 for i, ch in enumerate(self.chars)}  # +3 for special tokens
        self.stoi['<PAD>'] = PAD_TOKEN
        self.stoi['<START>'] = START_TOKEN
        self.stoi['<END>'] = END_TOKEN
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        print(f"Vocabulary size: {self.vocab_size}")

        # Process images and ALL captions (for random sampling)
        self.images = []
        self.all_captions = []  # List of lists: [[cap0, cap1, ...], [cap0, cap1, ...], ...]
        print("Processing images...")
        for item in tqdm(self.ds):
            img = item['image'].convert('RGB').resize((image_size, image_size))
            # Store all 5 captions for random sampling during training
            captions = [item[f'caption_{i}'].lower() for i in range(5)]
            self.images.append(img)
            self.all_captions.append(captions)

        print(f"Loaded {len(self.images)} images with 5 captions each")

    def encode(self, s):
        tokens = [START_TOKEN] + [self.stoi.get(c, PAD_TOKEN) for c in s] + [END_TOKEN]
        # Pad or truncate
        if len(tokens) > MAX_CAPTION_LEN:
            tokens = tokens[:MAX_CAPTION_LEN-1] + [END_TOKEN]
        else:
            tokens = tokens + [PAD_TOKEN] * (MAX_CAPTION_LEN - len(tokens))
        return tokens

    def decode(self, tokens):
        chars = []
        for t in tokens:
            if t == END_TOKEN:
                break
            if t not in (PAD_TOKEN, START_TOKEN):
                chars.append(self.itos.get(t, '?'))
        return ''.join(chars)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import random
        img = self.images[idx]
        # Convert to tensor: (C, H, W) normalized to [-1, 1]
        img_tensor = torch.tensor(list(img.getdata()), dtype=torch.float32)
        img_tensor = img_tensor.view(self.image_size, self.image_size, 3).permute(2, 0, 1)
        img_tensor = img_tensor / 127.5 - 1.0

        # Randomly select one of the 5 captions (data augmentation)
        caption = random.choice(self.all_captions[idx])
        caption_tokens = torch.tensor(self.encode(caption), dtype=torch.long)
        return img_tensor, caption_tokens


# --- Vision Encoder ---
class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""
    def __init__(self, image_size, patch_size, n_embd):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size  # RGB * patch pixels

        self.proj = nn.Linear(patch_dim, n_embd)
        self.pos_embd = nn.Parameter(torch.randn(1, self.n_patches, n_embd) * 0.02)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        p = self.patch_size

        # Reshape to patches: (B, n_patches, patch_dim)
        x = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5)  # (B, H/p, W/p, C, p, p)
        x = x.reshape(B, -1, C * p * p)  # (B, n_patches, patch_dim)

        x = self.proj(x) + self.pos_embd
        return x


class Head(nn.Module):
    """One head of self-attention"""
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
            # Cross-attention
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
    """Transformer block for vision encoder (no causal mask)"""
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
    """Transformer block for caption decoder (causal mask + cross-attention)"""
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

        # Vision encoder
        self.patch_embed = PatchEmbedding(image_size, patch_size, n_embd)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(n_embd, n_head) for _ in range(n_layer // 2)
        ])

        # Text decoder
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(max_caption_len, n_embd)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(n_embd, n_head, max_caption_len) for _ in range(n_layer // 2)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def encode_image(self, img):
        x = self.patch_embed(img)
        for block in self.encoder_blocks:
            x = block(x)
        return x

    def forward(self, img, caption_tokens, targets=None):
        # Encode image
        img_features = self.encode_image(img)

        # Decode caption
        B, T = caption_tokens.shape
        pos = torch.arange(T, device=caption_tokens.device)
        x = self.token_embed(caption_tokens) + self.pos_embed(pos)

        for block in self.decoder_blocks:
            x = block(x, img_features)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B*T, C),
                targets.reshape(B*T),
                ignore_index=PAD_TOKEN
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, img, start_token, end_token, max_len=32):
        self.eval()
        img_features = self.encode_image(img.unsqueeze(0))

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


# --- Training ---
def main():
    print("\n" + "="*50)
    print("Image Captioning Transformer (Flickr8k)")
    print("="*50)

    # Load dataset
    train_dataset = Flickr8kDataset('train', IMAGE_SIZE)
    val_dataset = Flickr8kDataset('validation', IMAGE_SIZE)
    val_dataset.stoi = train_dataset.stoi  # Share vocabulary
    val_dataset.itos = train_dataset.itos
    val_dataset.vocab_size = train_dataset.vocab_size

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = ImageCaptioningModel(
        vocab_size=train_dataset.vocab_size,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        max_caption_len=MAX_CAPTION_LEN
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Patches: {N_PATCHES} ({PATCH_SIZE}x{PATCH_SIZE} each)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    @torch.no_grad()
    def evaluate(loader, dataset):
        model.eval()
        total_loss = 0
        n_batches = 0
        for img, caption in loader:
            img = img.to(device)
            caption = caption.to(device)
            x = caption[:, :-1]
            y = caption[:, 1:]
            _, loss = model(img, x, y)
            total_loss += loss.item()
            n_batches += 1
        return total_loss / n_batches

    print("\nTraining...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for img, caption in pbar:
            img = img.to(device)
            caption = caption.to(device)

            # Input: all tokens except last, Target: all tokens except first
            x = caption[:, :-1]
            y = caption[:, 1:]

            logits, loss = model(img, x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss = total_loss / n_batches
        val_loss = evaluate(val_loader, val_dataset)
        print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Generate a sample caption
        model.eval()
        sample_img = train_dataset.images[0]
        img_tensor = torch.tensor(list(sample_img.getdata()), dtype=torch.float32)
        img_tensor = img_tensor.view(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1)
        img_tensor = (img_tensor / 127.5 - 1.0).to(device)

        tokens = model.generate(img_tensor, START_TOKEN, END_TOKEN)
        generated = train_dataset.decode(tokens)
        print(f"  Generated: '{generated}'")
        print(f"  Target:    '{train_dataset.all_captions[0][0]}'")

    print("\nTraining complete!")

    # Save model
    torch.save(model.state_dict(), 'image_caption_model.pth')
    print("Model saved to image_caption_model.pth")


if __name__ == "__main__":
    main()
