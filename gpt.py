from timeit import default_timer as timer

import torch
import torch.nn as nn
from torch.nn import functional as F

BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 200
N_EMBED = 384
N_LAYER = 6
N_HEAD = 6
DROPOUT = 0.2
PATH = "gpt_model"

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


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

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.s_attn(x).split(N_EMBED, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)  # (B, nh, T, T)
        wei = wei.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = wei @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.resi_dropout(self.proj(out))

        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
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
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.block = nn.Sequential(*[Block(n_head=N_HEAD) for _ in range(N_LAYER)])
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.block(x)
        logits = self.lm_head(x)

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

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel()
m = model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


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
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
