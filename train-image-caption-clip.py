"""
Image Captioning with Pretrained CLIP Vision Encoder

Improvement over train-image-caption.py:
- Uses CLIP ViT-B/32 pretrained vision encoder (frozen or fine-tuned)
- Only trains the decoder + projection layer
- Should achieve better results with less training

Uses Flickr8k dataset (~8000 images, 5 captions each)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import clip  # pip install git+https://github.com/openai/CLIP.git

# --- Device ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Hyperparameters ---
BATCH_SIZE = 16
IMAGE_SIZE = 224          # CLIP expects 224x224
CLIP_EMBD = 512           # CLIP ViT-B/32 output dimension
N_EMBD = 256              # Decoder embedding dimension (larger since encoder is stronger)
N_HEAD = 8
N_LAYER = 6               # More decoder layers
MAX_CAPTION_LEN = 48
LEARNING_RATE = 1e-4      # Lower LR for pretrained model
NUM_EPOCHS = 20
FREEZE_ENCODER = True     # Freeze CLIP encoder (faster training)

# --- Tokenization (character-level) ---
PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2

# --- Dataset ---
class Flickr8kDatasetCLIP(Dataset):
    def __init__(self, split='train', clip_preprocess=None):
        print(f"Loading Flickr8k {split} split...")
        self.ds = load_dataset('jxie/flickr8k', split=split)
        self.clip_preprocess = clip_preprocess

        # Build vocabulary from all captions
        print("Building vocabulary...")
        all_chars = set()
        for item in self.ds:
            for i in range(5):
                caption = item[f'caption_{i}']
                all_chars.update(caption.lower())

        self.chars = sorted(list(all_chars))
        self.stoi = {ch: i + 3 for i, ch in enumerate(self.chars)}
        self.stoi['<PAD>'] = PAD_TOKEN
        self.stoi['<START>'] = START_TOKEN
        self.stoi['<END>'] = END_TOKEN
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        print(f"Vocabulary size: {self.vocab_size}")

        # Store raw PIL images and captions (preprocess on-the-fly for CLIP)
        self.images = []
        self.all_captions = []
        print("Loading images...")
        for item in tqdm(self.ds):
            img = item['image'].convert('RGB')
            captions = [item[f'caption_{i}'].lower() for i in range(5)]
            self.images.append(img)
            self.all_captions.append(captions)

        print(f"Loaded {len(self.images)} images with 5 captions each")

    def encode(self, s):
        tokens = [START_TOKEN] + [self.stoi.get(c, PAD_TOKEN) for c in s] + [END_TOKEN]
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

        # Apply CLIP preprocessing
        img_tensor = self.clip_preprocess(img)

        # Randomly select one of the 5 captions
        caption = random.choice(self.all_captions[idx])
        caption_tokens = torch.tensor(self.encode(caption), dtype=torch.long)
        return img_tensor, caption_tokens


# --- Decoder Components ---
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


class CLIPCaptioningModel(nn.Module):
    def __init__(self, vocab_size, clip_model, clip_embd, n_embd, n_head, n_layer, max_caption_len, freeze_encoder=True):
        super().__init__()
        self.max_caption_len = max_caption_len
        self.clip_model = clip_model
        self.freeze_encoder = freeze_encoder

        # Freeze CLIP encoder if specified
        if freeze_encoder:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Project CLIP features to decoder dimension
        # CLIP ViT-B/32 outputs: (B, 50, 768) for grid features or (B, 512) for pooled
        # We'll use the visual transformer's patch embeddings (50 = 1 CLS + 49 patches for 224/32)
        self.visual_proj = nn.Linear(768, n_embd)  # CLIP ViT hidden dim is 768

        # Text decoder
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(max_caption_len, n_embd)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(n_embd, n_head, max_caption_len) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def encode_image(self, img):
        """Get CLIP visual features (patch embeddings, not just CLS token)"""
        with torch.set_grad_enabled(not self.freeze_encoder):
            # Get visual transformer
            visual = self.clip_model.visual

            # Manually extract patch embeddings from CLIP ViT
            x = visual.conv1(img.type(visual.conv1.weight.dtype))  # (B, 768, 7, 7)
            x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, 768, 49)
            x = x.permute(0, 2, 1)  # (B, 49, 768)

            # Add CLS token and positional embeddings
            cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls_token, x], dim=1)  # (B, 50, 768)
            x = x + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x)

            # Pass through transformer
            x = x.permute(1, 0, 2)  # (50, B, 768)
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)  # (B, 50, 768)

        # Project to decoder dimension
        x = self.visual_proj(x.float())  # (B, 50, n_embd)
        return x

    def forward(self, img, caption_tokens, targets=None):
        # Encode image with CLIP
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
    print("\n" + "="*60)
    print("Image Captioning with Pretrained CLIP Vision Encoder")
    print("="*60)

    # Load CLIP model
    print("\nLoading CLIP ViT-B/32...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    print(f"CLIP loaded on {device}")

    # Load dataset
    train_dataset = Flickr8kDatasetCLIP('train', clip_preprocess)
    val_dataset = Flickr8kDatasetCLIP('validation', clip_preprocess)
    val_dataset.stoi = train_dataset.stoi
    val_dataset.itos = train_dataset.itos
    val_dataset.vocab_size = train_dataset.vocab_size

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = CLIPCaptioningModel(
        vocab_size=train_dataset.vocab_size,
        clip_model=clip_model,
        clip_embd=CLIP_EMBD,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        max_caption_len=MAX_CAPTION_LEN,
        freeze_encoder=FREEZE_ENCODER
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"CLIP encoder frozen: {FREEZE_ENCODER}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE} (CLIP standard)")

    # Only optimize trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

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
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for img, caption in pbar:
            img = img.to(device)
            caption = caption.to(device)

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

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': train_dataset.stoi,
                'itos': train_dataset.itos,
            }, 'image_caption_clip_best.pth')
            print(f"  Saved best model (val_loss: {val_loss:.4f})")

        # Generate sample captions
        model.eval()
        sample_img = train_dataset.images[0]
        img_tensor = clip_preprocess(sample_img).to(device)

        tokens = model.generate(img_tensor, START_TOKEN, END_TOKEN)
        generated = train_dataset.decode(tokens)
        print(f"  Generated: '{generated}'")
        print(f"  Target:    '{train_dataset.all_captions[0][0]}'")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': train_dataset.stoi,
        'itos': train_dataset.itos,
    }, 'image_caption_clip_final.pth')
    print("Final model saved to image_caption_clip_final.pth")

    # Save vocabulary for export
    import json
    with open('vocab_clip.json', 'w') as f:
        json.dump({'stoi': train_dataset.stoi, 'itos': {str(k): v for k, v in train_dataset.itos.items()}}, f)
    print("Vocabulary saved to vocab_clip.json")


if __name__ == "__main__":
    main()
