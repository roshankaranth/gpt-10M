def get_stats(tokens, stats = None):
    stats = {} if stats is None else stats
    
    for pair in zip(tokens, tokens[1:]): 
        stats[pair] = stats.get(pair, 0) + 1

    return stats

def merge(tokens, pair, idx):
    newids = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            newids.append(idx)
            i+=2
        else:
            newids.append(tokens[i])
            i+=1

    return newids

class BasicTokenizer:
    
    def __init__(self):
        self.vocab = {k:bytes([k]) for k in range(256)}
        self.merges = {}

    def train(self, text, vocab_size):
        tokens = text.encode("utf-8") 
        ids = list(tokens)

        merges_count = max(0, vocab_size - 256)
        
        for i in range(merges_count):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)

            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        print(f"Training completed!\nNumber of merges : {len(self.merges)}\nVocab size : {len(self.vocab)}")

    def encode(self, text):
        tokens = text.encode("utf-8")
        tokens = list(tokens)

        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break

            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        text = b"".join(self.vocab[idx] for idx in ids)
        text = text.decode("utf-8", errors = 'replace') #Why errors = 'replace' ? 

        return text

