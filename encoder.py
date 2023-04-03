import json

def save_tokenized_vector(tokens, filename):
    with open(filename, "w") as f:
        json.dump(tokens, f)

def load_tokenized_vector(filename):
    with open(filename, "r") as f:
        tokens = json.load(f)
        return tokens

class Encoder:
    def __init__(self, tokens):
        self.longest_token_size = 0
        self.tokens_by_word = {}
        self.tokens_by_id = {}
        self._load(tokens)

    def _load(self, tokens):
        count = 0 
        for token in tokens:
            self.tokens_by_word[token] = count
            self.tokens_by_id[count] = token
            count += 1
            if len(token) > self.longest_token_size:
                self.longest_token_size = len(token)
        
    def encode(self, text):
        text_size = len(text)
        pos = 0
        part_size = self.longest_token_size
        result = []
        while pos < text_size:
            if pos + part_size > text_size:
                part_size = text_size - pos

            while part_size > 0:
                word = text[pos : pos+part_size]
                if word in self.tokens_by_word:
                    number = self.tokens_by_word[word]
                    result.append(number)
                    break
                part_size -= 1

            if part_size == 0:
                raise f"Invalid dictionary; token not found '{text[pos : pos+1]}'"
            pos += part_size
            part_size = self.longest_token_size
        return result
    
    def decode(self, data):
        result = []
        for token in data:
            if token not in self.tokens_by_id:
                raise f"Invalid data; token not found '{token}'"
            result.append(self.tokens_by_id[token])
        return ''.join(result)
