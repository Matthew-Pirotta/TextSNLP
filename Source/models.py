from collections import Counter
from enum import StrEnum, IntEnum
import math
import random
import pickle
import os


random.seed(71)

from pre_processing import *

class NGramType(IntEnum):
    UNIGRAM = 0
    BIGRAM = 1
    TRIGRAM = 2
    INTERPOLATION = -1

class LanguageModel(StrEnum):
    VANILLA = "Vanilla"
    LAPLACE = "LAPLACE"
    UNK = "UNK"

__all__ = [ "NGramType", "LanguageModel", "Model"]

class Model():
    #NOTE todo pass var for upperbound of n-grams)
    def __init__(self, model_type: LanguageModel):
        self.ngrams:list[Counter] = [Counter(), Counter(), Counter()]
        self.model_type = model_type
        self.prob_func = self.get_prob_function(model_type)
        self.vocabulary = set()

        self.total_tokens:list[int] = [0] * len(self.ngrams)
        self.lambdas:list[float] = [0.1, 0.3, 0.6]
        self.rare_threshold = 1000 #TODO set a proper value

    
    def train(self, train_sentences:Sentences):
        #print(f"Training {self.model_type} model...")

        #Reset
        for ngram in self.ngrams:
            ngram.clear()
        self.vocabulary.clear()

        #Replace Rare words with UNK if model selected
        if self.model_type == LanguageModel.UNK:
            word_counts = Counter(word for sentence in train_sentences for word in sentence)
            rare_words = set(word for word, count in word_counts.items() if count <= self.rare_threshold)

            processed_sentences = [
                [word if word not in rare_words else "<UNK>" for word in sentence]
                for sentence in train_sentences
            ]
        else:
            processed_sentences = train_sentences


        #Calc ngram for each sentence
        for sentence in processed_sentences:
            for i, ngram in enumerate(self.ngrams):
                n = i+1
                ngram.update(PreProcessing.generate_n_grams(sentence, n))
    
            self.vocabulary.update(word for word in sentence)

        #Total tokens for each ngram
        for i, ngram in enumerate(self.ngrams):
            self.total_tokens[i] = sum(ngram.values())

        print(f"{self.model_type} trained with {len(self.vocabulary)} unique words and {self.total_tokens} total tokens")

    def vanilla_ngram_prob(self, n_gram_type:NGramType, word, prev_words:NGram) -> float:
        n_gram = tuple(prev_words + (word,))

        # unigram
        if n_gram_type == n_gram_type.UNIGRAM:
            if self.total_tokens[0] == 0:
                return 0.0
            return self.ngrams[0][n_gram] / self.total_tokens[0]

        # High order n-grams (2+)
        n_minus_1_gram = tuple(prev_words)
        n_model =  self.ngrams[n_gram_type]
        n_minus_1_model = self.ngrams[n_gram_type-1]

        if (n_minus_1_gram not in n_minus_1_model) or (n_minus_1_model[n_minus_1_gram] == 0): return 0

        return n_model[n_gram] / n_minus_1_model[n_minus_1_gram]
    
    def laplace_ngram_prob(self, n_gram_type:NGramType, word, prev_words:NGram) -> float:
        n_gram = tuple(prev_words + (word,))
        vocab_size = len(self.vocabulary)

        # unigram
        if n_gram_type == n_gram_type.UNIGRAM:
            count = self.ngrams[0][n_gram]
            total = self.total_tokens[0]
            return (count+1) / (total + vocab_size)

        # High order n-grams (2+)
        n_minus_1_gram = tuple(prev_words)
        n_model =  self.ngrams[n_gram_type]
        n_minus_1_model = self.ngrams[n_gram_type-1]

        count = n_model[n_gram]
        prev_count = n_minus_1_model[n_minus_1_gram]
        return (count + 1) / (prev_count + vocab_size)
    
    def unk_ngram_prob(self, n_gram_type:NGramType, word, prev_words:NGram) -> float:
        word = word if word in self.vocabulary else "<UNK>"
        prev_words = tuple(w if w in self.vocabulary else "<UNK>" for w in prev_words)
        return self.vanilla_ngram_prob(n_gram_type, word, prev_words)

    def _interpolation_prob(self, word, prev_words:NGram) -> float:
        """lambdas - [unigram, bigram, trigram, ...]"""
        highest_n = 1 + len(prev_words)

        total_prob = 0
        for i in range(highest_n):
            n_gram_type = NGramType(i)
            context = prev_words[-i:] if i > 0 else ()  # last i words
            prob = self.prob_func(n_gram_type, word, context)
            total_prob += self.lambdas[i] * prob

        return total_prob

    def calc_sen_probability(self, sentence:Sentence, n_gram_type:NGramType) -> float:
        #P(w1, w2, ..., wn) = P(wi| wn-t,wn-t)
        log_prob = 0.0

        start_index = 1 if n_gram_type == NGramType.INTERPOLATION else n_gram_type 

        for i in range(start_index, len(sentence)):
            word = sentence[i]

            if n_gram_type == NGramType.INTERPOLATION:
                # Use full context up to trigram (2 previous words)
                prev_words = tuple(sentence[max(0, i - 2):i])
                word_prob = self._interpolation_prob(word, prev_words)
            else:
                prev_words = tuple(sentence[i-n_gram_type:i])
                word_prob = self.prob_func(n_gram_type, word, prev_words)

            if word_prob > 0:
                log_prob += math.log(word_prob)
            else:
                log_prob += math.log(1e-10)  # Assign a small probability to unseen n-grams

        
        return log_prob

    def calc_perplexity(self, test_sentences:Sentences, n_gram_type:NGramType) -> float:
        total_log_prob = 0.0
        total_words = 0

        for sentence in test_sentences:
            total_log_prob += self.calc_sen_probability(sentence, n_gram_type)
            total_words += len(sentence)
        
        if total_words == 0:
            return float('inf')
        
        avg_log_prob = total_log_prob / total_words
        perplexity = math.exp(-avg_log_prob)
        return perplexity
    
    def get_prob_function(self, model_type: LanguageModel):
        mapping = {
            LanguageModel.VANILLA: self.vanilla_ngram_prob,
            LanguageModel.LAPLACE: self.laplace_ngram_prob,
            LanguageModel.UNK: self.unk_ngram_prob
        }

        return mapping[model_type]
            
    def generate_next_word(self, sentence:Sentence, n_gram_type:NGramType) -> str:
        # Randomly choose a word if no context is available
        if not sentence:
            return random.choice(list(self.vocabulary))  

        words:list[str] = []
        weights:list[float] = []

        window_previous_words = max(0, n_gram_type - 1)
        prev_words = tuple(sentence[-window_previous_words:])
        for word in self.vocabulary:
            #Start token should not be generated
            if word == "<s>": continue

            prob = self.prob_func(n_gram_type, word, prev_words)
            if prob <= 0: continue 
            words.append(word)
            weights.append(prob)

        # Return end-of-sentence token if no valid words are found,
        # a randomly generated word would not make sense in the context of the sentence
        if not words:
          print("No valid words were found to be generated")
          return "</s>"  

        chosen_word = random.choices(words, weights, k=1)[0]
        return chosen_word

    def generate_sentence(self, start_of_sentence:str, n_gram_type:NGramType, max_length:int = 22) -> str:
        #Pre process sentence into expected shape
        processed_start_of_sentence = ["<s>"] + start_of_sentence.split()

        if self.model_type == LanguageModel.UNK:
            processed_start_of_sentence = [w if w in self.vocabulary else "<UNK>" for w in processed_start_of_sentence]

        final_sentence:list[str] = processed_start_of_sentence[:]
        generated_word = ""
        while (len(final_sentence) < max_length) and not generated_word == "</s>": #sentence ends in </s>:
            generated_word = self.generate_next_word(final_sentence, n_gram_type)
            final_sentence.append(generated_word)

        return " ".join(final_sentence[1:-1])
    
    def save_model(self, folder_path:str):
        os.makedirs(os.path.dirname(folder_path), exist_ok=True)

        fileDir = f"{folder_path}/{self.model_type}"
        with open(fileDir, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
