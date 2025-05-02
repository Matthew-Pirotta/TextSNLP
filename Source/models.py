from collections import Counter
from enum import Enum
from abc import ABC, abstractmethod

from pre_processing import PreProcessing, Sentence, Sentences

class NGramType(Enum):
    #TODO see actual value encoding
    #TODO NOTE iterpolation -1?
    UNIGRAM = 0
    BIGRAM = 1
    TRIGRAM = 2

class Model(ABC):
    def __init__(self, name: str):
        self.name = name
        self.ngrams:list[Counter] = [Counter(), Counter(), Counter()]
        self.vocabulary = set()
        self.total_tokens:list = []
    
    def train(self, train_sentences):
        print(f"Training {self.name} model...")

        #Reset
        for ngram in self.ngrams:
            ngram.clear()
        self.vocabulary.clear()

        #Calc ngram for each sentence
        for sentence in train_sentences:
            for i, ngram in enumerate(self.ngrams):
                n = i+1
                ngram.update(PreProcessing.generate_n_gram(sentence, n))
    
            self.vocabulary.update(word for word in sentence)
            print(f"Model trained with {len(self.vocabulary)} unique words and {self.total_tokens} total tokens")


        #Total tokens for each ngram
        for i, ngram in enumerate(self.ngrams):
            self.total_tokens[i] = sum(ngram.values())

    def vanilla_ngram_prob(self, NGramType:NGramType, word, *prev_words) -> float:
        n_gram = tuple(prev_words + (word,))

        # unigram
        if NGramType == NGramType.UNIGRAM:
            total_unigrams = sum(self.ngrams[0].values())
            if total_unigrams == 0:
                return 0.0
            return self.ngrams[0][n_gram] / total_unigrams

        # High order n-grams (2+)
        n_minus_1_gram = tuple(prev_words)
        n_model =  self.ngrams[NGramType.value]
        n_minus_1_model = self.ngrams[NGramType.value-1]

        if (n_minus_1_gram not in n_minus_1_model) or (n_minus_1_model[n_minus_1_gram] == 0): return 0

        return n_model[n_gram] / n_minus_1_model[n_minus_1_gram]


    def calc_sentence_prob(self, sentence:Sentence, model_type:NGramType, prob_func):
        #P(w1, w2, ..., wn) = P(wi| wn-t,wn-t)

        word_prob = prob_func()


        match model_type:
            case NGramType.UNIGRAM:
                pass
            case NGramType.BIGRAM:
                pass
            case NGramType.TRIGRAM:
                pass

        #= P(w1|<s>) * P(w2|<s>,w1) * P(w3|w1,w2) * ... * P(wn|wn-2,wn-1)

    def calc_perplexity(self):
        pass
