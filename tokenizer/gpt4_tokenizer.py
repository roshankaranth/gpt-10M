import tiktoken
from tokenizer.regexTokenizer import RegexTokenizer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

def bpe(mergeable_ranks, token, max_rank = None):
    
    parts = [bytes([b]) for b in token]

    while True:
        min_rank = None
        min_idx = None

        for index, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = index
                min_rank = rank

        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break

        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx+1]] + parts[min_idx + 2:]

    return parts

            
def recover_merges(mergeable_ranks):
    merges = {}

    for token, rank in mergeable_ranks.items():
        if len(token) < 2: continue
        pair = tuple(bpe(mergeable_ranks, token, rank))
        assert len(pair) == 2

        merges[(mergeable_ranks[pair[0]] , mergeable_ranks[pair[1]])] = rank

    return merges 

class GPT4Tokenizer(RegexTokenizer):
    
    def __init__(self):
        super().__init__(pattern = GPT4_SPLIT_PATTERN)

        enc = tiktoken.get_encoding("cl100k_base")
        mergable_ranks = enc._mergeable_ranks

        self.merges = recover_merges(mergable_ranks)

        vocab = {idx : bytes([idx]) for idx in range(256)}
        for (p0,p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        self.vocab = vocab

        self.byte_shuffle = {i: mergable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k,v in self.byte_shuffle.items()}

        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def decode(self, ids):

        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        text = text_bytes.decode("utf-8", errors="replace")

        return text
    
    def _encode_chunk(self, text_bytes):
        
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        ids = super()._encode_chunk(text_bytes)
        return ids
