"""
Image Captioning with CLIP Vision Encoder + GloVe Word Embeddings

Combines two pretrained components:
1. CLIP ViT-B/32 for image encoding (frozen)
2. GloVe 6B 100d for word embeddings (frozen or fine-tuned)

This tests whether pretrained text embeddings help like pretrained vision did.

Download GloVe first:
  wget https://nlp.stanford.edu/data/glove.6B.zip
  unzip glove.6B.zip
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import clip
import numpy as np
import os

# --- Device ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Hyperparameters ---
BATCH_SIZE = 16
IMAGE_SIZE = 224              # CLIP expects 224x224
CLIP_EMBD = 512               # CLIP ViT-B/32 output dimension
GLOVE_DIM = 100               # GloVe embedding dimension (using 100d version)
N_EMBD = 256                  # Decoder hidden dimension
N_HEAD = 8
N_LAYER = 6
MAX_CAPTION_LEN = 20          # Word-level: 20 words (vs 48 characters)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
FREEZE_ENCODER = True         # Freeze CLIP encoder
FREEZE_GLOVE = False          # Fine-tune GloVe embeddings (can experiment with True)
MIN_WORD_FREQ = 2             # Minimum word frequency to include in vocab

# --- Tokenization (word-level) ---
PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3

# --- GloVe Loader ---
def load_glove_embeddings(glove_path, word2idx, embedding_dim=100):
    """Load GloVe embeddings for words in vocabulary"""
    print(f"Loading GloVe embeddings from {glove_path}...")

    # Initialize with random embeddings
    vocab_size = len(word2idx)
    embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01

    # Special tokens get zero vectors
    embeddings[PAD_TOKEN] = 0
    embeddings[START_TOKEN] = np.random.randn(embedding_dim) * 0.1
    embeddings[END_TOKEN] = np.random.randn(embedding_dim) * 0.1
    embeddings[UNK_TOKEN] = np.random.randn(embedding_dim) * 0.1

    found = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe"):
            parts = line.strip().split()
            word = parts[0]
            if word in word2idx:
                idx = word2idx[word]
                embeddings[idx] = np.array(parts[1:], dtype=np.float32)
                found += 1

    print(f"Found {found}/{vocab_size-4} words in GloVe ({100*found/(vocab_size-4):.1f}%)")
    return torch.tensor(embeddings, dtype=torch.float32)


# --- Dataset ---
class Flickr8kDatasetCLIPGloVe(Dataset):
    def __init__(self, split='train', clip_preprocess=None, word2idx=None, idx2word=None):
        print(f"Loading Flickr8k {split} split...")
        self.ds = load_dataset('jxie/flickr8k', split=split)
        self.clip_preprocess = clip_preprocess

        # Build vocabulary from all captions (word-level)
        if word2idx is None:
            print("Building word-level vocabulary...")
            word_counts = {}
            for item in self.ds:
                for i in range(5):
                    caption = item[f'caption_{i}'].lower()
                    # Simple tokenization: split on whitespace, keep punctuation attached
                    words = caption.split()
                    for word in words:
                        word_counts[word] = word_counts.get(word, 0) + 1

            # Filter by frequency
            self.words = sorted([w for w, c in word_counts.items() if c >= MIN_WORD_FREQ])
            self.word2idx = {w: i + 4 for i, w in enumerate(self.words)}  # Reserve 0-3 for special tokens
            self.word2idx['<PAD>'] = PAD_TOKEN
            self.word2idx['<START>'] = START_TOKEN
            self.word2idx['<END>'] = END_TOKEN
            self.word2idx['<UNK>'] = UNK_TOKEN
            self.idx2word = {i: w for w, i in self.word2idx.items()}
        else:
            self.word2idx = word2idx
            self.idx2word = idx2word
            self.words = [w for w in word2idx.keys() if w not in ['<PAD>', '<START>', '<END>', '<UNK>']]

        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary size: {self.vocab_size} words")

        # Store raw PIL images and captions
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
        """Convert sentence to word indices"""
        words = s.split()
        tokens = [START_TOKEN] + [self.word2idx.get(w, UNK_TOKEN) for w in words] + [END_TOKEN]
        if len(tokens) > MAX_CAPTION_LEN:
            tokens = tokens[:MAX_CAPTION_LEN-1] + [END_TOKEN]
        else:
            tokens = tokens + [PAD_TOKEN] * (MAX_CAPTION_LEN - len(tokens))
        return tokens

    def decode(self, tokens):
        """Convert word indices to sentence"""
        words = []
        for t in tokens:
            if t == END_TOKEN:
                break
            if t not in (PAD_TOKEN, START_TOKEN):
                words.append(self.idx2word.get(t, '<UNK>'))
        return ' '.join(words)

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


class CLIPGloVeCaptioningModel(nn.Module):
    """
    Image captioning with:
    - CLIP ViT-B/32 vision encoder (pretrained, frozen)
    - GloVe word embeddings (pretrained, optionally fine-tuned)
    - Transformer decoder (trained from scratch)
    """
    def __init__(self, vocab_size, clip_model, glove_embeddings, n_embd, n_head, n_layer,
                 max_caption_len, freeze_encoder=True, freeze_glove=False):
        super().__init__()
        self.max_caption_len = max_caption_len
        self.clip_model = clip_model
        self.freeze_encoder = freeze_encoder

        # Freeze CLIP encoder
        if freeze_encoder:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Project CLIP features to decoder dimension (768 -> n_embd)
        self.visual_proj = nn.Linear(768, n_embd)

        # GloVe word embeddings
        glove_dim = glove_embeddings.shape[1]
        self.token_embed = nn.Embedding.from_pretrained(glove_embeddings, freeze=freeze_glove)

        # Project GloVe dim to decoder dim if different
        self.glove_proj = nn.Linear(glove_dim, n_embd) if glove_dim != n_embd else nn.Identity()

        # Positional embeddings (learned, much shorter for word-level)
        self.pos_embed = nn.Embedding(max_caption_len, n_embd)

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(n_embd, n_head, max_caption_len) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def encode_image(self, img):
        """Get CLIP visual features (patch embeddings)"""
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

    def forward(self, img, caption_tokens, targets=None):
        # Encode image with CLIP
        img_features = self.encode_image(img)

        # Decode caption
        B, T = caption_tokens.shape
        pos = torch.arange(T, device=caption_tokens.device)

        # GloVe embeddings -> project to decoder dim
        tok_emb = self.token_embed(caption_tokens)
        tok_emb = self.glove_proj(tok_emb)
        x = tok_emb + self.pos_embed(pos)

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

            tok_emb = self.token_embed(x)
            tok_emb = self.glove_proj(tok_emb)
            x = tok_emb + self.pos_embed(pos)

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
    print("Image Captioning: CLIP Vision + GloVe Word Embeddings")
    print("="*60)

    # Check for GloVe file
    glove_path = "glove.6B.100d.txt"
    if not os.path.exists(glove_path):
        print(f"\nERROR: GloVe file not found at {glove_path}")
        print("Please download GloVe embeddings:")
        print("  wget https://nlp.stanford.edu/data/glove.6B.zip")
        print("  unzip glove.6B.zip")
        return

    # Load CLIP model
    print("\nLoading CLIP ViT-B/32...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    print(f"CLIP loaded on {device}")

    # Load dataset (word-level)
    train_dataset = Flickr8kDatasetCLIPGloVe('train', clip_preprocess)
    val_dataset = Flickr8kDatasetCLIPGloVe('validation', clip_preprocess,
                                            train_dataset.word2idx, train_dataset.idx2word)

    # Load GloVe embeddings
    glove_embeddings = load_glove_embeddings(glove_path, train_dataset.word2idx, GLOVE_DIM)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = CLIPGloVeCaptioningModel(
        vocab_size=train_dataset.vocab_size,
        clip_model=clip_model,
        glove_embeddings=glove_embeddings,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        max_caption_len=MAX_CAPTION_LEN,
        freeze_encoder=FREEZE_ENCODER,
        freeze_glove=FREEZE_GLOVE
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    clip_params = sum(p.numel() for p in clip_model.parameters())
    glove_params = train_dataset.vocab_size * GLOVE_DIM

    print(f"\n{'='*40}")
    print("Model Configuration:")
    print(f"{'='*40}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"CLIP encoder (frozen): {clip_params:,}")
    print(f"GloVe embeddings ({'frozen' if FREEZE_GLOVE else 'fine-tuned'}): {glove_params:,}")
    print(f"Vocabulary size: {train_dataset.vocab_size} words")
    print(f"Max caption length: {MAX_CAPTION_LEN} words")
    print(f"Training samples: {len(train_dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    @torch.no_grad()
    def evaluate(loader):
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
        val_loss = evaluate(val_loader)
        print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'word2idx': train_dataset.word2idx,
                'idx2word': train_dataset.idx2word,
            }, 'image_caption_clip_glove_best.pth')
            print(f"  Saved best model (val_loss: {val_loss:.4f})")

        # Generate sample caption
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
        'word2idx': train_dataset.word2idx,
        'idx2word': train_dataset.idx2word,
    }, 'image_caption_clip_glove_final.pth')
    print("Model saved to image_caption_clip_glove_final.pth")

    # Save vocabulary
    import json
    with open('vocab_clip_glove.json', 'w') as f:
        json.dump({
            'word2idx': train_dataset.word2idx,
            'idx2word': {str(k): v for k, v in train_dataset.idx2word.items()}
        }, f)
    print("Vocabulary saved to vocab_clip_glove.json")


if __name__ == "__main__":
    main()
