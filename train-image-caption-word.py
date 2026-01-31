"""
Image Captioning Transformer with Word-Level Tokenization

Improvement over train-image-caption.py (character-level):
- Word-level tokenization instead of character-level
- Shorter sequences (15 words vs 48 characters)
- Larger vocabulary (~2500 words vs ~70 characters)
- Should train faster due to shorter sequences

Parameter count comparison:
- Character-level: ~980K params
  - Token embedding: 70 × 128 = 8,960
  - Position embedding: 48 × 128 = 6,144
  - Output layer: 128 × 70 = 8,960

- Word-level: ~1.7M params (+750K)
  - Token embedding: 2500 × 128 = 320,000 (+311K)
  - Position embedding: 20 × 128 = 2,560 (-3.5K)
  - Output layer: 128 × 2500 = 320,000 (+311K)
  - Most params in embeddings, not transformer blocks

Uses Flickr8k dataset (~8000 images, 5 captions each)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import re
from collections import Counter

# --- Device ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Hyperparameters ---
BATCH_SIZE = 16
IMAGE_SIZE = 128
PATCH_SIZE = 8
N_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 256
N_EMBD = 128
N_HEAD = 4
HEAD_SIZE = N_EMBD // N_HEAD
N_LAYER = 4
MAX_CAPTION_LEN = 20  # Words, not characters! Much shorter
LEARNING_RATE = 3e-4
NUM_EPOCHS = 30
MIN_WORD_FREQ = 2  # Words appearing less than this are <UNK>

# --- Special Tokens ---
PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3

# --- Dataset with Word-Level Tokenization ---
class Flickr8kWordDataset(Dataset):
    def __init__(self, split='train', image_size=64, vocab=None):
        print(f"Loading Flickr8k {split} split...")
        self.ds = load_dataset('jxie/flickr8k', split=split)
        self.image_size = image_size

        # Build vocabulary from all captions (only for training set)
        print("Building word vocabulary...")
        word_counts = Counter()
        for item in self.ds:
            for i in range(5):
                caption = item[f'caption_{i}'].lower()
                words = self.tokenize(caption)
                word_counts.update(words)

        if vocab is None:
            # Build vocab from scratch (training set)
            # Filter by minimum frequency
            filtered_words = [w for w, c in word_counts.items() if c >= MIN_WORD_FREQ]
            filtered_words = sorted(filtered_words)

            self.stoi = {
                '<PAD>': PAD_TOKEN,
                '<START>': START_TOKEN,
                '<END>': END_TOKEN,
                '<UNK>': UNK_TOKEN,
            }
            for i, word in enumerate(filtered_words):
                self.stoi[word] = i + 4  # +4 for special tokens
        else:
            # Use provided vocab (for validation set)
            self.stoi = vocab

        self.itos = {i: w for w, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        print(f"Vocabulary size: {self.vocab_size} words")
        print(f"Total unique words in dataset: {len(word_counts)}")
        print(f"Words filtered (freq < {MIN_WORD_FREQ}): {len(word_counts) - self.vocab_size + 4}")

        # Process images and captions
        self.images = []
        self.all_captions = []
        print("Processing images...")
        for item in tqdm(self.ds):
            img = item['image'].convert('RGB').resize((image_size, image_size))
            captions = [item[f'caption_{i}'].lower() for i in range(5)]
            self.images.append(img)
            self.all_captions.append(captions)

        print(f"Loaded {len(self.images)} images with 5 captions each")

        # Show sample tokenization
        sample_caption = self.all_captions[0][0]
        sample_tokens = self.encode(sample_caption)
        print(f"\nSample tokenization:")
        print(f"  Caption: '{sample_caption}'")
        print(f"  Words: {self.tokenize(sample_caption)}")
        print(f"  Token IDs: {sample_tokens[:10]}... (len={len(sample_tokens)})")

    def tokenize(self, text):
        """Simple word tokenization: split on whitespace and punctuation"""
        # Remove special characters, keep only words and basic punctuation
        text = text.lower()
        # Split on whitespace, keep punctuation as separate tokens
        words = re.findall(r"[\w]+|[.,!?;']", text)
        return words

    def encode(self, caption):
        """Convert caption string to token IDs"""
        words = self.tokenize(caption)
        tokens = [START_TOKEN]
        for word in words:
            tokens.append(self.stoi.get(word, UNK_TOKEN))
        tokens.append(END_TOKEN)

        # Pad or truncate
        if len(tokens) > MAX_CAPTION_LEN:
            tokens = tokens[:MAX_CAPTION_LEN - 1] + [END_TOKEN]
        else:
            tokens = tokens + [PAD_TOKEN] * (MAX_CAPTION_LEN - len(tokens))
        return tokens

    def decode(self, tokens):
        """Convert token IDs back to string"""
        words = []
        for t in tokens:
            if t == END_TOKEN:
                break
            if t not in (PAD_TOKEN, START_TOKEN):
                word = self.itos.get(t, '<UNK>')
                words.append(word)
        return ' '.join(words)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import random
        img = self.images[idx]
        # Convert to tensor: (C, H, W) normalized to [-1, 1]
        img_tensor = torch.tensor(list(img.getdata()), dtype=torch.float32)
        img_tensor = img_tensor.view(self.image_size, self.image_size, 3).permute(2, 0, 1)
        img_tensor = img_tensor / 127.5 - 1.0

        # Randomly select one of the 5 captions
        caption = random.choice(self.all_captions[idx])
        caption_tokens = torch.tensor(self.encode(caption), dtype=torch.long)
        return img_tensor, caption_tokens


# --- Vision Encoder (same as character-level) ---
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
        return wei @ v


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

        # Vision encoder
        self.patch_embed = PatchEmbedding(image_size, patch_size, n_embd)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(n_embd, n_head) for _ in range(n_layer // 2)
        ])

        # Text decoder - larger embedding for word-level vocab
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
        img_features = self.encode_image(img)

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
    def generate(self, img, start_token, end_token, max_len=20):
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
    print("Image Captioning Transformer with Word-Level Tokenization")
    print("="*60)

    # Load dataset
    train_dataset = Flickr8kWordDataset('train', IMAGE_SIZE)
    val_dataset = Flickr8kWordDataset('validation', IMAGE_SIZE, vocab=train_dataset.stoi)
    val_dataset.itos = train_dataset.itos

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

    # Count parameters by component
    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*50}")
    print("Parameter Count Breakdown:")
    print(f"{'='*50}")
    print(f"Token embedding:     {count_params(model.token_embed):,} ({count_params(model.token_embed)/n_params*100:.1f}%)")
    print(f"Position embedding:  {count_params(model.pos_embed):,} ({count_params(model.pos_embed)/n_params*100:.1f}%)")
    print(f"Patch embedding:     {count_params(model.patch_embed):,}")
    print(f"Encoder blocks:      {sum(count_params(b) for b in model.encoder_blocks):,}")
    print(f"Decoder blocks:      {sum(count_params(b) for b in model.decoder_blocks):,}")
    print(f"Output layer:        {count_params(model.lm_head):,} ({count_params(model.lm_head)/n_params*100:.1f}%)")
    print(f"{'='*50}")
    print(f"TOTAL:               {n_params:,}")
    print(f"{'='*50}")

    print(f"\nComparison with character-level (~980K):")
    print(f"  Difference: +{n_params - 980000:,} params")
    print(f"  Ratio: {n_params / 980000:.2f}x")

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max caption length: {MAX_CAPTION_LEN} words (vs 48 chars)")
    print(f"Vocabulary size: {train_dataset.vocab_size} (vs ~70 chars)")

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
            torch.save(model.state_dict(), 'image_caption_word_best.pth')

        # Generate sample captions
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
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), 'image_caption_word_final.pth')
    print("Model saved to image_caption_word_final.pth")

    # Save vocabulary
    import json
    with open('vocab_word.json', 'w') as f:
        json.dump({'stoi': train_dataset.stoi, 'itos': {str(k): v for k, v in train_dataset.itos.items()}}, f)
    print("Vocabulary saved to vocab_word.json")


if __name__ == "__main__":
    main()
