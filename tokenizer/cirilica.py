from .tokenizer import Tokenizer, count_conseq_tokens, merge
import regex as re
from tqdm import tqdm


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class Cirilica(Tokenizer):
    def __init__(self, pattern=None):
        """
            Токенизатор за српску ћирилицу
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.dict_size = 0
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def _create_vocabulary(self, train_text:str):
        chars = sorted(list(set(train_text)))
        self.vocab = { i:ch for i,ch in enumerate(chars) }


    # Тренирање токенајзера
    # Свака епоха замени један токен као најчешће понављани пар
    def train(self, text, vocab_size):
        """
            Прави вокабулар токена на основу датог тренинг текста
        """
        # Направи почетни вокабулар
        self._create_vocabulary(text)
        self.dict_size = len(self.vocab)
        self.vocab_size = vocab_size

        # Енкодер за превођење текста у број
        self.encoder = {v:k for k, v in self.vocab.items()}
        
        # Подели на делове са регексом
        text_chunked = re.findall(self.compiled_pattern, text)

        # Преведи текст у почетне токене
        tokens = []
        for chunk in text_chunked:
            tokens.append([self.encoder[slovo] for slovo in chunk])

        # За сваки регекс изброј counts понављања
        i = len(self.vocab.keys())
        steps = vocab_size - i  # укупан број корака

        # Користи tqdm за праћење напретка
        for _ in tqdm(range(steps), desc="Training BPE"):
            counts = {}

            # Извући број понављања
            for chunk in tokens:
                counts = count_conseq_tokens(chunk, counts)
            
            # Извлачи највећи пар
            pair = max(counts, key=counts.get)

            # Замени најбројнији пар
            tokens = [merge(chunk, pair, i) for chunk in tokens]

            self.vocab[i] = (self.vocab[pair[0]] , self.vocab[pair[1]])
            
            # Повећај бројач
            i += 1

        # Додај специјалне токене

    def decode(self, tokens: list[int]) -> str:
        """
            Дешифрује дату листу токена назад у ћирилични текст
        """
        text:str = ""
        
        # Претварање токена од највећег ка најмањем из речника [vocab_size] -> [0]
        i = self.vocab_size - 1
        reverse = {v:k for k, v in self.vocab.items()}
 
        while i >= self.dict_size:
            pair = self.vocab[i]
            j = 0

            while j < len(tokens):
                if (tokens[j]) == i:
                    # Мало лудачка линија али ради посао, избаци токен из листе и дода 2 нова
                    tokens = tokens[:j] + [reverse[pair[0]]] + [reverse[pair[1]]] + ([] if j==len(tokens)-1 else tokens[j+1:])
                j+=1
            i-=1

        # Претвори остатак токена у слова
        for c in tokens:
            text = text + self.vocab[c]

        return text



    def encode(self, text: str) -> list[int]:
        """
            Претвара дати ћирилични текст у листу токена
        """
        tokens = []

        # Подели на делове са регексом
        text_chunked = re.findall(self.compiled_pattern, text)

        # Преведи текст у почетне токене
        for chunk in text_chunked:
            tokens.append([self.encoder[slovo] for slovo in chunk])

        i = self.dict_size
        while i < self.vocab_size:
            # Извлачи највећи пар
            pair = self.vocab[i]
 
            # Замени пар
            tokens = [merge(chunk, pair, i) for chunk in tokens]

            # Повећај бројач
            i+=1

        # Изравна токене
        tokens = [token for token_chunk in tokens for token in token_chunk]
        return tokens