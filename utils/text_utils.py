import torch
import os
import re
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        # Special tokens
        self.add_word('<pad>')   # 0
        self.add_word('<start>') # 1
        self.add_word('<end>')   # 2
        self.add_word('<unk>')   # 3

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def __len__(self):
        return len(self.word2idx)

def clean_text(text):
    """[FIX 1] Removes punctuation and makes lowercase to prevent 'word.' != 'word'"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Strips commas, periods, quotes, etc.
    return text

def build_vocab(annotation_root_dir, min_freq=1):
    print(f"Building vocabulary from {annotation_root_dir}...")
    vocab = Vocabulary()
    counter = Counter()
    
    # [FIX 2] SORTED recursive scan to guarantee file read order
    for root, dirs, files in sorted(os.walk(annotation_root_dir)):
        for file in sorted(files):
            if file.endswith(".ant"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                text = clean_text(parts[1])
                                counter.update(text.split())
                except Exception as e:
                    print(f"Skipping {file}: {e}")

    # [FIX 3] SORTED dictionary insertion so word indices NEVER change
    # We sort alphabetically by the word string (x[0])
    for word, count in sorted(counter.items(), key=lambda x: x[0]):
        if count >= min_freq:
            vocab.add_word(word)
            
    print(f"Vocabulary size: {len(vocab)}")
    return vocab

def text_to_indices(text, vocab, max_len=20):
    text = clean_text(text) # Ensure input text is clean before tokenizing
    tokens = text.split()
    indices = [vocab('<start>')] + [vocab(token) for token in tokens] + [vocab('<end>')]
    
    if len(indices) < max_len:
        indices += [vocab('<pad>')] * (max_len - len(indices))
    else:
        indices = indices[:max_len-1] + [vocab('<end>')]
        
    return torch.tensor(indices, dtype=torch.long)