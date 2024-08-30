import re
from collections import defaultdict, Counter

class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2id = {}
        self.id2word = {}
        self.bpe_ranks = {}
        self.vocab = {}

    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)'+ bigram + r'(?!\S)')
        new_vocab = {}
        for word in vocab:
            new_word = p.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def build_vocab(self, corpus):
        # Start with character-level vocabulary
        vocab = defaultdict(int)
        for word in corpus:
            word = ' '.join(list(word))+ ' </w>'
            vocab[word] += 1

        while len(self.word2id) < self.vocab_size:
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)

            # Adding the new subword to the vocabulary
            new_token = ''.join(best)
            if new_token not in self.word2id:
                self.word2id[new_token] = len(self.word2id)
                self.id2word[len(self.id2word) - 1] = new_token
                self.bpe_ranks[best] = len(self.bpe_ranks)


        # Include the final vocabulary with unique IDs
        for word in vocab:
            token = ''.join(word.split())
            if token not in self.word2id:
                self.word2id[token] = len(self.word2id)
                self.id2word[len(self.word2id) - 1] = token
    
    # tokenize one word
    def tokenize(self, text):
        text = ' '.join(list(text))+' </w>'
        tokens = text.split()

        while True:
            pairs = self.get_stats({text: 1})
            if not pairs:
                break
            
            # Im not sure !
            # best = min(pairs, key=self.bpe_ranks.get)
            best = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
           
            if best not in self.bpe_ranks:
                break

            # Merge
            tokens = self.merge_vocab(best, {' '.join(tokens): 1}).split()

        return tokens
            # new_text = ' '.join(self.merge_vocab(best, {text: 1}))
            # # check if its possible
            # if new_text == text:
            #     break
            # text = new_text
            # tokens = text.split()
        
        # return tokens
            

    # def encode(self, text):
    #     token = self.to
