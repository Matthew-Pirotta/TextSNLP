from collections import Counter
from enum import IntEnum
import math

from pre_processing import *

#TODO convert to int enum
class NGramType(IntEnum):
    #TODO see actual value encoding
    #TODO NOTE iterpolation -1?
    UNIGRAM = 0
    BIGRAM = 1
    TRIGRAM = 2
 
class Model():
    #NOTE todo pass var for upperbound of n-grams)
    def __init__(self, name: str):
        self.name = name
        self.ngrams:list[Counter] = [Counter(), Counter(), Counter()]
        self.vocabulary = set()
        self.total_tokens:list[int] = [0] * len(self.ngrams)
    
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
                ngram.update(PreProcessing.generate_n_grams(sentence, n))
    
            self.vocabulary.update(word for word in sentence)


        #Total tokens for each ngram
        for i, ngram in enumerate(self.ngrams):
            self.total_tokens[i] = sum(ngram.values())

        print(f"Model trained with {len(self.vocabulary)} unique words and {self.total_tokens} total tokens")

    def vanilla_ngram_prob(self, n_gram_type:NGramType, word, prev_words:NGram) -> float:
        n_gram = tuple(prev_words + (word,))

        # unigram
        if n_gram_type == n_gram_type.UNIGRAM:
            total_unigrams = sum(self.ngrams[0].values())
            if total_unigrams == 0:
                return 0.0
            return self.ngrams[0][n_gram] / total_unigrams

        # High order n-grams (2+)
        n_minus_1_gram = tuple(prev_words)
        n_model =  self.ngrams[n_gram_type]
        n_minus_1_model = self.ngrams[n_gram_type-1]

        if (n_minus_1_gram not in n_minus_1_model) or (n_minus_1_model[n_minus_1_gram] == 0): return 0

        return n_model[n_gram] / n_minus_1_model[n_minus_1_gram]
    
    def interpolation_prob(self, word, prev_words:NGram, lambdas:list[float] = [0.1, 0.3, 0.6]) -> float:
        """lambdas - [unigram, bigram, trigram, ...]"""
        highest_n = 1 + len(prev_words)

        total_prob = 0
        for i in range(highest_n):
            n_gram_type = NGramType(i)
            context = prev_words[-i:] if i > 0 else ()  # last i words
            prob += self.vanilla_ngram_prob(n_gram_type, word, context)
            total_prob += lambdas[i] * prob

        return total_prob

    def calc_n_gram_sent_prob(self, sentence:Sentence, n_gram_type:NGramType, prob_func) -> float:
        #P(w1, w2, ..., wn) = P(wi| wn-t,wn-t)
        log_prob = 0.0

        for i in range(n_gram_type, len(sentence)):
            word = sentence[i]
            prev_words = tuple(sentence[i-n_gram_type:i])
            word_prob = prob_func(n_gram_type, word, prev_words)

            if word_prob > 0:
                log_prob += math.log(word_prob)
            else:
                log_prob += -float('inf')
        
        return log_prob

    #TODO IDK #Until model has a window of its n, a lower order n is used
    def interpolation_logic(self):
        """
        for i in range(1, len(sentence)):
            n = min(i, model_type)
            word = sentence[i]
            #TODO NOTE possible bug when (i-n) is less than 0 but i dont think its possible. IM pretty sure its yapping hard
            prev_words = sentence[i-n:i]
            """

    def calc_perplexity(self, test_sentences:Sentences, n_gram_type:NGramType, prob_func) -> float:
        total_log_prob = 0.0
        total_words = 0

        for sentence in test_sentences:
            total_log_prob += self.calc_n_gram_sent_prob(sentence, n_gram_type, prob_func)
            total_words += len(sentence)
        
        if total_words == 0:
            return float('inf')
        
        avg_log_prob = total_log_prob / total_words
        perplexity = math.exp(-avg_log_prob)
        return perplexity