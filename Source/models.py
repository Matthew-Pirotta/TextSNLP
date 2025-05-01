from collections import Counter

from pre_processing import PreProcessing

class Model:
    def __init__(self, name: str):
        self.name = name
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.vocabulary = set()
        self.total_tokens = 0
    
    def train(self, train_sentences):
        print(f"Training {self.name} model...")

        #Reset
        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.trigram_counts.clear()
        self.vocabulary.clear()

        for sentence in train_sentences:
            unigrams = PreProcessing.generate_n_gram(sentence, 1)
            self.unigram_counts.update(unigrams)

            bigrams = PreProcessing.generate_n_gram(sentence, 2)
            self.bigram_counts.update(bigrams)

            trigrams = PreProcessing.generate_n_gram(sentence, 3)
            self.trigram_counts.update(trigrams)

            self.vocabulary.update(word for word in sentence)
