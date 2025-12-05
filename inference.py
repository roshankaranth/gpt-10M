import torch
import torch.nn as nn
from torch.nn import functional as F
from timeit import default_timer as timer

from tokenizer.regexTokenizer import RegexTokenizer

BLOCK_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EMBED = 384
N_LAYER = 6
N_HEAD = 6
DROPOUT = 0.2
MODEL_PATH = "model_v1"
FILE_PATH = "output.txt"
VOCAB_SIZE = 256
PATTERN = r"[A-Za-z]+(?:'[A-Za-z]+)?|[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]"
MAX_NEW_TOKENS = 10000
MAX_SEQ_LEN = BLOCK_SIZE + MAX_NEW_TOKENS

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = RegexTokenizer()
tokenizer.train(VOCAB_SIZE, text)
print(f"RegexTokenizer training complete!")

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

def precompute_rope_embeddings(head_dim, max_seq_len, base=10000):

    inv_freq = 1.0/(base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    emb = freqs.repeat_interleave(2, dim=1)
    
    return emb

def apply_rotary_pos_emb(x, rope_emb, offset = 0):

    B, nh, T, hs = x.shape

    rope_emb_sliced = rope_emb[offset: offset + T, :]

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
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view( 1, 1, BLOCK_SIZE, BLOCK_SIZE))
        self.n_head = num_heads
        self.proj = nn.Linear(N_EMBED, N_EMBED, bias=False)
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resi_dropout = nn.Dropout(DROPOUT)
        head_dim = N_EMBED // num_heads
        self.register_buffer("rope_embeds", precompute_rope_embeddings(head_dim, MAX_SEQ_LEN),persistent=False)

    def forward(self, x, past_kv = None):
        B, T, C = x.shape #BATCH_SIZE, BLOCK_SIZE, N_EMBED | (during generation (1, 1, N_EMBED))

        q, k, v = self.s_attn(x).split(N_EMBED, dim=2) #(B,T,C) * (C, 3*C) -> (B,T,3*C) -> (B,T,C) , (B,T,C) , (B,T,C) | (during generation, (1,1,C) * (C, 3*C) -> (1,1,3*C) -> (1,1,C), (1,1,C), (1,1,C))
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, T, nh, hs) -> (B, nh, T, hs) | (1, 1, nh, hs) -> (1, nh, 1, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, hs) | (1, 1, nh, hs) -> (1, nh, 1, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, hs) | (1, 1, nh, hs) -> (1, nh, 1, hs)

        if past_kv is None:
            past_len = 0
        else:
            past_k, past_v = past_kv
            past_len = past_k.shape[-2]

        q = apply_rotary_pos_emb(q, self.rope_embeds, offset=past_len)
        k = apply_rotary_pos_emb(k, self.rope_embeds, offset=past_len)

        if past_kv is not None:
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

            if k.shape[-2] > BLOCK_SIZE:
                k = k[:, :, -BLOCK_SIZE:, :]
                v = v[:, :, -BLOCK_SIZE:, :]


        T_k = k.shape[-2]

        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)  # (B, nh, T, hs) @ (B, nh, hs, T) --> (B, nh, T, T)/(root of head_size) | (1, nh, 1, hs) @ (1, nh, hs, new_T) --> (1, nh, 1, new_T)
        wei = wei.masked_fill(self.tril[:, :, T_k  - T:T_k , :T_k ] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = wei @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C) #(B, T, nh, hs) -> (B,T,C)

        out = self.resi_dropout(self.proj(out))

        return out, (k,v)


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

    def forward(self, x, past_kv = None):
        attn_out, past_kv = self.sa(self.ln1(x), past_kv) #residual path + layer-norm before attention block
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x)) #residual path + layer norm before feed forward network
        return x,past_kv    


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBED) #Each unique token will have an embedding of size N_EMBED
        # self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED) #Similarly each position in a block, will have an embedding vector of size N_EMBED 
        # self.block = nn.Sequential(*[Block(n_head=N_HEAD) for _ in range(N_LAYER)]) #* is used to convert list of block to positional arguments.
        self.blocks = nn.ModuleList([Block(n_head=N_HEAD) for _ in range(N_LAYER)])
        self.lm_head = nn.Linear(N_EMBED, VOCAB_SIZE) #(BLOCK_SIZE * N_EMBED) * (N_EMBED * VOCAB_SIZE)

    def forward(self, idx, targets=None, kv_cache = None):
        B, T = idx.shape #BATCH_SIZE, BLOCK_SIZE. (inputs)

        tok_emb = self.token_embedding_table(idx) #(B,T,C) where C is N_EMBED
        # pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) #(T,C) matrix
        x = tok_emb#(B,T,C) + (T,C) (broadcasting happens here)
        #x = self.block(x) #Through Transformer (B, T, C) -> (B, T, C)
        if kv_cache is None:
            for block in self.blocks:
                x,_ = block(x)
        else:
            for i, block in enumerate(self.blocks):
                x, kv_cache[i] = block(x,kv_cache[i])
        
        logits = self.lm_head(x) #Language Model head (B, T, C) -> (B, T, VOCAB_SIZE)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss, kv_cache

    def generate(self, idx, max_new_tokens):
        
        kv_cache = [None for _ in range(N_LAYER)]

        logits,_,kv_cache = self(idx, kv_cache = kv_cache) #prefilling the cache
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -BLOCK_SIZE:]
            # idx_cond = idx[:, -1:]

            logits,_,kv_cache = self(idx_cond, kv_cache = None)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx



model = GPT()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

time_start = timer()
with torch.no_grad():
    model_output = tokenizer.decode(model.generate(context, max_new_tokens=MAX_NEW_TOKENS)[0].tolist())

time_end = timer()

with open(FILE_PATH, "w") as file:
    file.write(model_output)

print(f"Time for generating response : {time_end - time_start} secs")
