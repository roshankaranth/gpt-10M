import regex as re
from tokenizer.base import BasicTokenizer, get_stats, merge

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
#The split pattern is so that, merges don't happen across different categories (letters, numbers, punctuations)
class RegexTokenizer(BasicTokenizer):

    def __init__(self, pattern = None):
        super().__init__()

        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, vocab_size, text):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text)

        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        
        merges = {}
        vocab = {idx : bytes([idx]) for idx in range(256)}

        for i in range(num_merges):

            stats = {}
            for id in ids: #get stats from all the chunks created
                stats = get_stats(id, stats)

            pair = max(stats, key=stats.get)

            idx = 256 + i
            ids = [merge(id, pair, idx) for id in ids] #In all chunks replace the pair with new token

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab

    def _encode_chunk(self, text_bytes):
        tokens = list(text_bytes)

        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break

            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens
        
    def encode_ordinary(self, text):

        text_chunks = re.findall(self.compiled_pattern, text)

        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            tokens = self._encode_chunk(chunk_bytes)
            ids.extend(tokens)

        return ids
    
    def register_special_tokens(self, special_tokens):
        #special tokens is string -> int
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v:k for k,v in self.special_tokens.items()}

    def decode(self, ids):
        #ids is a list of integers, returns pythons string

        part_bytes = []

        for id in ids:
            if id in self.vocab:
                part_bytes.append(self.vocab[id])
            elif id in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[id].encode("utf-8"))
                #special inverse token is map from int to string. Need to convert string to bytes
            else:
                raise ValueError(f"invalid token if : {id}")
            
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")

        return text
    
    def encode(self, text, allowed_special = "none_raise"):

        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        
        if not special:
            return self.encode_ordinary(text)
        
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))

        return ids
    




    
    



    

