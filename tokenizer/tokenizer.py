
#   ПОМОЋНЕ ФУНКЦИЈЕ


# ---------------------------------------------------------
# Помоћна функција која броји понаљање узастопних токена

def count_conseq_tokens(ids, counts={}):
    """
    За дату листу интиџера, врати речник броја понављања узаступних парова
    Пример: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Опционо ажурира већ дата пребројавања
    """
    for pair in zip(ids[:-1], ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

# ---------------------------------------------------------
# Функција која обацује нове токене у текст

def merge(ids, pair, idx):
    """
    Замени све парове pair који се појављују у ids листи са idx
    Пример: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    new_ids = []
    i = 0
    ids_n = len(ids)
    if ids_n <= 1 or not pair:
        return ids
    while i < ids_n - 1:
        if (ids[i], ids[i+1]) == pair:
            new_ids.append(idx)
            i+=1
            
        else:
            new_ids.append(ids[i])
        if i == ids_n-2:
            new_ids.append(ids[i+1]) 
        i+=1
        
    return new_ids


# ---------------------------------------------------------
# Апстрактна Токенајзер класа

class Tokenizer:
    """Основна класа за Токејзере"""

    def __init__(self):
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = {} # int -> char

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError
    
    # TODO: FINISH THIS
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges.items():
                f.write(f"{idx1} {idx2}\n")
            # vocab
            f.write(f"vocab\n")
            for idx1, idx2 in self.vocab:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        # vocab_file = file_prefix + ".vocab"
        # inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        # with open(vocab_file, "w", encoding="utf-8") as f:
        #     for idx, token in self.vocab.items():
  

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

            # read the vocan
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()