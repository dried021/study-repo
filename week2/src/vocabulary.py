import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json
import os
from typing import List, Dict, Tuple


class RNNTokenizer:
    '''
    참고: https://medium.com/@nibniw/building-a-large-language-model-from-scratch-a-comprehensive-technical-guide-eb2f4478663c
    '''
    def __init__(self, vocab_size: int = 10000, min_freq: int = 2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<sos>" , 3: "<eos>"}

    def build_vocab(self, sentences: List[str]):
        word_counts = Counter()
        for sentence in sentences:
            words = sentence.lower().split()
            word_counts.update(words)
        
        for word, count in word_counts.most_common(self.vocab_size - len(self.word2idx)):
            if count >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, sentence: str, max_len: int = None) -> List[int]:
        words = sentence.lower().split()
        tokens = [self.word2idx.get(word, self.word2idx["<unk>"]) for word in words]
        tokens = [self.word2idx["<sos>"]]+tokens+[self.word2idx["<eos>"]]

        if max_len is not None and len(tokens) > max_len:
            tokens = tokens[:max_len-1] + [self.word2idx["<eos>"]]

        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        return " ".join(self.idx2word.get(token, "<unk>") for token in tokens)
    
    def save_vocab(self, filepath: str):
        vocab = {"word2idx": self.word2idx, "idx2word": self.idx2word}

        with open(filepath, 'w') as f:
            json.dump(vocab, f)

    def load_vocab(self, filepath: str):
        with open(filepath, 'r') as f:
            vocab = json.load(f)
            self.word2idx = vocab["word2idx"]
            self.idx2word = {int(k): v for k, v in vocab["idx2word"].items()}
            
