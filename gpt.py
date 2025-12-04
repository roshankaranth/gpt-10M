from timeit import default_timer as timer

import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer.regexTokenizer import RegexTokenizer

BATCH_SIZE = 64 #In one batch there are, 64*256 samples.
BLOCK_SIZE = 256 #context window is 256
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 200
N_EMBED = 384 #Embedding dimension
N_LAYER = 6 
N_HEAD = 6  #Number of Attention heads
DROPOUT = 0.2
PATH = "model_v2"
VOCAB_SIZE = 256 #Vocab size of tokenizer
PATTERN = r"[A-Za-z]+(?:'[A-Za-z]+)?|[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]"

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = RegexTokenizer()
tokenizer.train(VOCAB_SIZE, text)
print(f"RegexTokenizer training complete!")

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

n = int(0.9 * len(data)) #size of training set
train_data = data[:n]
val_data = data[n:] #validation set


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,)) #get a list of size BATCH_SIZE, with random integers from 0 to (len(data) - BLOCK_SIZE)(NOT INCLUSIVE)
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix]) #matrix of size BATCH_SIZE * BLOCK_SIZE
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix]) #matrix of size BATCH_SIZE * BLOCK_SIZE
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

def precompute_rope_embeddings(head_dim, max_seq_len, base=10000):

    inv_freq = 1.0/(base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    emb = freqs.repeat_interleave(2, dim=1)
    
    return emb

def apply_rotary_pos_emb(x, rope_emb):

    B, nh, T, hs = x.shape
    seq_len = T
    rope_emb_sliced = rope_emb[:seq_len, :]

    emb = rope_emb_sliced.unsqueeze(0).unsqueeze(0)

    cos = emb.cos()
    sin = emb.sin()

    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x_partner = torch.stack([-x_reshaped[...,1], x_reshaped[...,0]], dim=-1)
    x_partner = x_partner.flatten(-2)

    return (x * cos + x_partner * sin).type_as(x)

class MultipleHeadSelfAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.s_attn = nn.Linear(N_EMBED, 3 * N_EMBED, bias=False) 
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(
                1, 1, BLOCK_SIZE, BLOCK_SIZE
            ),
        )
        self.n_head = num_heads
        self.proj = nn.Linear(N_EMBED, N_EMBED, bias=False)
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resi_dropout = nn.Dropout(DROPOUT)
        head_dim = N_EMBED // num_heads
        self.register_buffer("rope_embeds", precompute_rope_embeddings(head_dim, BLOCK_SIZE),persistent=False)

    def forward(self, x):
        B, T, C = x.shape #BATCH_SIZE, BLOCK_SIZE, N_EMBED | (during generation (1, 1, N_EMBED))

        q, k, v = self.s_attn(x).split(N_EMBED, dim=2) #(B,T,C) * (C, 3*C) -> (B,T,3*C) -> (B,T,C) , (B,T,C) , (B,T,C) | (during generation, (1,1,C) * (C, 3*C) -> (1,1,3*C) -> (1,1,C), (1,1,C), (1,1,C))
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, T, nh, hs) -> (B, nh, T, hs) | (1, 1, nh, hs) -> (1, nh, 1, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, hs) | (1, 1, nh, hs) -> (1, nh, 1, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, hs) | (1, 1, nh, hs) -> (1, nh, 1, hs)

        q = apply_rotary_pos_emb(q, self.rope_embeds)
        k = apply_rotary_pos_emb(k, self.rope_embeds)

        T_k = k.shape[-2]

        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)  # (B, nh, T, hs) @ (B, nh, hs, T) --> (B, nh, T, T)/(root of head_size) | (1, nh, 1, hs) @ (1, nh, hs, new_T) --> (1, nh, 1, new_T)
        wei = wei.masked_fill(self.tril[:, :, T_k  - T:T_k , :T_k ] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = wei @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C) #(B, T, nh, hs) -> (B,T,C)

        out = self.resi_dropout(self.proj(out))

        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), #(B,T,C) * (C,4*C) -> (B, T, 4*C)
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), #(B, T, 4*C) * (4*C, C) -> (B,T,C)
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_head):
        super().__init__()
        self.sa = MultipleHeadSelfAttention(n_head)
        self.ffwd = FeedForward(N_EMBED)
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ln2 = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #residual path + layer-norm before attention block
        x = x + self.ffwd(self.ln2(x)) #residual path + layer norm before feed forward network
        return x    


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBED) #Each unique token will have an embedding of size N_EMBED
        # self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED) #Similarly each position in a block, will have an embedding vector of size N_EMBED 
        self.block = nn.Sequential(*[Block(n_head=N_HEAD) for _ in range(N_LAYER)]) #* is used to convert list of block to positional arguments.
        self.lm_head = nn.Linear(N_EMBED, VOCAB_SIZE) #(BLOCK_SIZE * N_EMBED) * (N_EMBED * VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape #BATCH_SIZE, BLOCK_SIZE. (inputs)

        tok_emb = self.token_embedding_table(idx) #(B,T,C) where C is N_EMBED
        # pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) #(T,C) matrix
        x = tok_emb#(B,T,C) + (T,C) (broadcasting happens here)
        x = self.block(x) #Through Transformer (B, T, C) -> (B, T, C)
        logits = self.lm_head(x) #Language Model head (B, T, C) -> (B, T, VOCAB_SIZE)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = GPT()
m = model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


time_start = timer()

for iter in range(MAX_ITERS):
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

time_end = timer()

print(f"Time to train model : {time_end - time_start} seconds")

torch.save(m.state_dict(), PATH)

context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
