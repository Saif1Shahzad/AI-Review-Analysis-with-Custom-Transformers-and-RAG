
import os
import gzip
import json
import math
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Constants and configuration
# ----------------------------
SEED = 42
MAX_SEQ_LEN = 128
DEC_SRC_LEN = 80
MAX_GEN_LEN = 40

DATA_FILES = {
    "beauty": "data/beauty.json.gz",
    "cellphones": "data/cellphones.json.gz",
    "electronics": "data/electronics.json.gz",
    "home": "data/home.json.gz",
    "sports": "data/sports.json.gz",
}
SELECTED_CATEGORIES = ["sports", "cellphones", "electronics"]
REVIEWS_PER_CATEGORY = 12000

ENC_D_MODEL = 128
ENC_NUM_HEADS = 4
ENC_NUM_LAYERS = 3
ENC_D_FF = 256
ENC_DROPOUT = 0.2

DEC_D_MODEL = 128
DEC_NUM_HEADS = 4
DEC_NUM_LAYERS = 3
DEC_D_FF = 256
DEC_DROPOUT = 0.2
DEC_MAX_LEN = DEC_SRC_LEN + MAX_GEN_LEN + 5

SENT_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}
LEN_LABELS = {0: "Short", 1: "Medium", 2: "Long"}
SENT_WORDS = {0: "negative", 1: "neutral", 2: "positive"}
LEN_WORDS = {0: "brief", 1: "moderate", 2: "detailed"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="NLP Assignment 3 Demo", layout="wide")
st.title("Transformer + RAG Review Explainer")
st.write("Runs the trained encoder, retrieval, and decoder using notebook outputs.")

# ----------------------------
# Data and preprocessing
# ----------------------------
def load_amazon_reviews(filepath, category, max_reviews=12000):
    reviews = []
    with gzip.open(filepath, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_reviews * 3:
                break
            try:
                obj = json.loads(line.strip())
                text = obj.get("reviewText", "") or obj.get("summary", "")
                rating = obj.get("overall", None)
                if text and rating and len(text.split()) >= 5:
                    reviews.append({
                        "text": text,
                        "rating": int(float(rating)),
                        "category": category,
                    })
                    if len(reviews) >= max_reviews:
                        break
            except Exception:
                continue
    return reviews

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    text = re.sub(r"(n't|'re|'ve|'ll|'d|'m|'s)", r" \1", text)
    return text.split()

def rating_to_sentiment(r):
    if r <= 2:
        return 0
    if r == 3:
        return 1
    return 2

def length_category(tokens):
    n = len(tokens)
    if n < 20:
        return 0
    if n <= 80:
        return 1
    return 2

@st.cache_data
def load_train_df():
    all_reviews = []
    for cat in SELECTED_CATEGORIES:
        path = DATA_FILES.get(cat, "")
        if not os.path.exists(path):
            continue
        all_reviews.extend(load_amazon_reviews(path, cat, REVIEWS_PER_CATEGORY))
    if not all_reviews:
        return None
    df = pd.DataFrame(all_reviews)
    df["clean_text"] = df["text"].apply(clean_text)
    df["tokens"] = df["clean_text"].apply(tokenize)
    df["sentiment"] = df["rating"].apply(rating_to_sentiment)
    df["length_cat"] = df["tokens"].apply(length_category)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n = len(df)
    n_train = int(0.70 * n)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    return train_df

@st.cache_data
def load_vocab():
    if not os.path.exists("results/vocab.pkl"):
        return None
    with open("results/vocab.pkl", "rb") as f:
        return pickle.load(f)

def build_decoder_vocab(word2idx):
    dec_word2idx = dict(word2idx)
    for tok in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
        if tok not in dec_word2idx:
            dec_word2idx[tok] = len(dec_word2idx)
    for tok in [
        "<SEP>",
        "__sent_negative__", "__sent_neutral__", "__sent_positive__",
        "__len_brief__", "__len_moderate__", "__len_detailed__",
    ]:
        if tok not in dec_word2idx:
            dec_word2idx[tok] = len(dec_word2idx)
    dec_idx2word = {v: k for k, v in dec_word2idx.items()}
    return dec_word2idx, dec_idx2word

# ----------------------------
# Model definitions
# ----------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.scale = math.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, V), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))
        x, _ = self.attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.W_O(x)

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=3,
                 d_ff=256, max_seq_len=MAX_SEQ_LEN, dropout=0.1,
                 pad_idx=0, num_sentiment=3, num_length_cat=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len + 10, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.pool_norm = nn.LayerNorm(d_model)
        self.sentiment_head = nn.Linear(d_model, num_sentiment)
        self.length_cat_head = nn.Linear(d_model, num_length_cat)
        self.pad_idx = pad_idx

    def make_src_mask(self, src):
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def encode(self, src):
        mask = self.make_src_mask(src)
        x = self.embedding(src)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        valid = (src != self.pad_idx).float().unsqueeze(-1)
        pooled = (x * valid).sum(1) / valid.sum(1).clamp(min=1e-9)
        return self.pool_norm(pooled)

    def forward(self, src):
        rep = self.encode(src)
        sent_logits = self.sentiment_head(rep)
        length_logits = self.length_cat_head(rep)
        return sent_logits, length_logits, rep

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        attn_out = self.masked_attn(x, x, x, causal_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=3,
                 d_ff=256, max_len=120, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len + 10, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.pad_idx = pad_idx

    def make_causal_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        return mask.to(DEVICE)

    def forward(self, x):
        S = x.size(1)
        causal_mask = self.make_causal_mask(S)
        x = self.embedding(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, causal_mask)
        x = self.norm_out(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids, eos_idx, max_new_tokens=MAX_GEN_LEN, temperature=0.8, top_k=40):
        self.eval()
        ids = prompt_ids.clone().to(DEVICE)
        for _ in range(max_new_tokens):
            logits = self.forward(ids)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                top_vals, _ = torch.topk(next_logits, top_k)
                threshold = top_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_tok], dim=1)
            if next_tok.item() == eos_idx:
                break
        return ids[0].tolist()

# ----------------------------
# Helpers for inference
# ----------------------------
def encode_tokens(tokens, word2idx, pad_idx, unk_idx):
    ids = [word2idx.get(t, unk_idx) for t in tokens[:MAX_SEQ_LEN]]
    ids += [pad_idx] * (MAX_SEQ_LEN - len(ids))
    return ids

def retrieve_top_k(query_embedding, train_embs_norm, k=3):
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
    sims = train_embs_norm @ q_norm
    top_k = np.argsort(sims)[::-1][:k]
    return top_k, sims[top_k]

def build_prompt_ids(tokens, sent_idx, len_idx, retrieved_rows, dec_word2idx, bos_idx, eos_idx, pad_idx, k):
    sent_tok = f"__sent_{SENT_WORDS[sent_idx]}__"
    len_tok = f"__len_{LEN_WORDS[len_idx]}__"
    parts = [sent_tok, len_tok] + tokens[:30]
    for rrow in retrieved_rows[:k]:
        parts += ["<SEP>"] + rrow["tokens"][:15]
    ids = [bos_idx] + [dec_word2idx.get(t, dec_word2idx["<UNK>"]) for t in parts[:DEC_SRC_LEN - 2]] + [eos_idx]
    ids = ids[:DEC_SRC_LEN] + [pad_idx] * (DEC_SRC_LEN - len(ids[:DEC_SRC_LEN]))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

def ids_to_text(ids, idx2word_map):
    tokens = []
    for i in ids:
        tok = idx2word_map.get(i, "<UNK>")
        if tok in ["<EOS>", "<PAD>"]:
            break
        if tok != "<BOS>":
            tokens.append(tok)
    return " ".join(tokens)

# ----------------------------
# Asset loading
# ----------------------------
@st.cache_resource
def load_models_and_data():
    vocab = load_vocab()
    if vocab is None:
        return None
    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]
    pad_idx = word2idx.get("<PAD>", 0)
    unk_idx = word2idx.get("<UNK>", 1)

    dec_word2idx, dec_idx2word = build_decoder_vocab(word2idx)
    bos_idx = dec_word2idx["<BOS>"]
    eos_idx = dec_word2idx["<EOS>"]
    dec_pad_idx = dec_word2idx["<PAD>"]

    encoder = TransformerEncoder(
        vocab_size=len(word2idx),
        d_model=ENC_D_MODEL,
        num_heads=ENC_NUM_HEADS,
        num_layers=ENC_NUM_LAYERS,
        d_ff=ENC_D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=ENC_DROPOUT,
        pad_idx=pad_idx,
    ).to(DEVICE)
    encoder.load_state_dict(torch.load("models/encoder_best.pt", map_location=DEVICE))
    encoder.eval()

    decoder = TransformerDecoder(
        vocab_size=len(dec_word2idx),
        d_model=DEC_D_MODEL,
        num_heads=DEC_NUM_HEADS,
        num_layers=DEC_NUM_LAYERS,
        d_ff=DEC_D_FF,
        max_len=DEC_MAX_LEN,
        dropout=DEC_DROPOUT,
        pad_idx=dec_pad_idx,
    ).to(DEVICE)
    decoder.load_state_dict(torch.load("models/decoder_best.pt", map_location=DEVICE))
    decoder.eval()

    if not os.path.exists("results/train_embeddings.npy"):
        return None
    train_embeddings = np.load("results/train_embeddings.npy", mmap_mode="r")
    norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    train_embeddings_norm = train_embeddings / (norms + 1e-9)

    train_df = load_train_df()
    return {
        "word2idx": word2idx,
        "idx2word": idx2word,
        "pad_idx": pad_idx,
        "unk_idx": unk_idx,
        "dec_word2idx": dec_word2idx,
        "dec_idx2word": dec_idx2word,
        "bos_idx": bos_idx,
        "eos_idx": eos_idx,
        "dec_pad_idx": dec_pad_idx,
        "encoder": encoder,
        "decoder": decoder,
        "train_embeddings_norm": train_embeddings_norm,
        "train_df": train_df,
    }

assets = {
    "results/vocab.pkl": os.path.exists("results/vocab.pkl"),
    "results/train_embeddings.npy": os.path.exists("results/train_embeddings.npy"),
    "models/encoder_best.pt": os.path.exists("models/encoder_best.pt"),
    "models/decoder_best.pt": os.path.exists("models/decoder_best.pt"),
}
st.sidebar.header("Assets Check")
for k, v in assets.items():
    st.sidebar.write(f"{k}: {'OK' if v else 'MISSING'}")

review = st.text_area(
    "Enter Amazon Review",
    height=180,
    placeholder="Battery died after two days. Very disappointed.",
)
k = st.sidebar.slider("Top-k Retrieved Reviews", 1, 10, 3)

if st.button("Analyze"):
    if not review.strip():
        st.warning("Enter a review first.")
    elif not all(assets.values()):
        st.error("Missing files. Run the notebook cells to generate models and embeddings.")
    else:
        bundle = load_models_and_data()
        if bundle is None:
            st.error("Unable to load assets or training data.")
        else:
            encoder = bundle["encoder"]
            decoder = bundle["decoder"]
            word2idx = bundle["word2idx"]
            dec_word2idx = bundle["dec_word2idx"]
            dec_idx2word = bundle["dec_idx2word"]
            pad_idx = bundle["pad_idx"]
            unk_idx = bundle["unk_idx"]
            bos_idx = bundle["bos_idx"]
            eos_idx = bundle["eos_idx"]
            dec_pad_idx = bundle["dec_pad_idx"]
            train_embeddings_norm = bundle["train_embeddings_norm"]
            train_df = bundle["train_df"]

            tokens = tokenize(clean_text(review))
            if not tokens:
                st.warning("Review is too short after cleaning.")
                st.stop()

            ids = encode_tokens(tokens, word2idx, pad_idx, unk_idx)
            ids_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                sent_logits, len_logits, rep = encoder(ids_tensor)
            sent_idx = int(torch.argmax(sent_logits, dim=1).item())
            len_idx = int(torch.argmax(len_logits, dim=1).item())
            sentiment = SENT_LABELS[sent_idx]
            length_cat = LEN_LABELS[len_idx]

            st.subheader("Predictions")
            c1, c2 = st.columns(2)
            c1.metric("Sentiment", sentiment)
            c2.metric("Length Category", length_cat)

            retrieved_rows = []
            if train_df is not None and len(train_df) == train_embeddings_norm.shape[0]:
                q_emb = rep.cpu().numpy()[0]
                top_k_idx, scores = retrieve_top_k(q_emb, train_embeddings_norm, k=k)
                retrieved_rows = [train_df.iloc[i] for i in top_k_idx]
                st.subheader("Retrieved Similar Reviews")
                for rank, (idx, sim) in enumerate(zip(top_k_idx, scores), start=1):
                    row = train_df.iloc[idx]
                    header = f"#{rank} (sim={sim:.3f})"
                    with st.expander(header, expanded=False):
                        st.write(row["clean_text"])
            else:
                st.warning("Training data not available for retrieval. Skipping RAG context.")

            prompt_ids = build_prompt_ids(
                tokens,
                sent_idx,
                len_idx,
                retrieved_rows,
                dec_word2idx,
                bos_idx,
                eos_idx,
                dec_pad_idx,
                k,
            )
            generated_ids = decoder.generate(prompt_ids, eos_idx, max_new_tokens=MAX_GEN_LEN)
            gen_text = ids_to_text(generated_ids, dec_idx2word)

            st.subheader("Generated Explanation")
            st.success(gen_text or "No explanation generated.")

st.caption("Tip: run the notebook first to create models/, results/, and vocab.pkl.")
