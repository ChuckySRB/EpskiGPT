import ast

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
        self.vocab = {} # int -> char | tuple
        self.encoder = {} # char -> int
        self.vocab_size = 0 # lenght of hole vocabulary
        self.dict_size = 0 # lenght of encoder

    def train(self, text, vocab_size):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError
    
    # TODO: FINISH THIS
    def save(self, file_path):
        """
            Чува истренирани вокабулар у .модел фајл
        """
        # write the model: to be used in load() later
        model_file = file_path + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("cirilo v1\n")
            f.write(f"{self.pattern}\n")
            # vocab
            f.write(f"{self.dict_size}\n")
            for i in range(self.dict_size):
                value = ord(self.vocab[i])
                f.write(f"{value}\n")
            # merges
            f.write(f"{self.vocab_size}\n")
            for i in range(self.dict_size, self.vocab_size):
                value = self.vocab[i]
                f.write(f"{value}\n")
        # write the vocab: for the human to look at
        # vocab_file = file_prefix + ".vocab"
        # inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        # with open(vocab_file, "w", encoding="utf-8") as f:
        #     for idx, token in self.vocab.items():
  

    def load(self, model_file):
        """
            Учитава сачувани модел
        """
        assert model_file.endswith(".model")
        # read the model file
        with open(model_file, 'r') as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            
            # read the pattern
            self.pattern = f.readline().strip()
            
            # read the vocab
            self.dict_size = int(f.readline().strip())
            for i in range(self.dict_size):
                value = chr(f.readline().strip())
                self.vocab[i] = value
            
            # merges
            self.vocab_size = f.readline().strip()
            for i in range(self.dict_size, self.vocab_size):
                value = ast.literal_eval(f.readline().strip())
                self.vocab[i] = value
            
