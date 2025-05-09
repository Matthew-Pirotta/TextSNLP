from collections import Counter
from enum import StrEnum, IntEnum

import math
import sys
import random
import pickle
import json
import os
from pympler import asizeof

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
    def __init__(self, model_type: LanguageModel):
        self.ngrams:list[Counter] = [Counter(), Counter(), Counter()]
        self.model_type = model_type
        self.prob_func = self.get_prob_function(model_type)
        self.vocabulary = set()

        self.total_tokens:list[int] = [0] * len(self.ngrams)
        self.lambdas:list[float] = [0.1, 0.3, 0.6] #Unigram, Bigram, Trigram
        self.threshold = 2

    def train(self, train_sentences:Sentences):
        #print(f"Training {self.model_type} model...")

        #Reset
        for ngram in self.ngrams:
            ngram.clear()
        self.vocabulary.clear()

        #Replace Rare words with UNK if model selected
        if self.model_type == LanguageModel.UNK:
            word_counts = Counter(word for sentence in train_sentences for word in sentence)
            rare_words = set(word for word, count in word_counts.items() if count <= self.threshold)

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

        is_missing_n_minus_1gram = n_minus_1_gram not in n_minus_1_model
        is_zero_count = n_minus_1_model[n_minus_1_gram] == 0
        if is_missing_n_minus_1gram or is_zero_count: return 0

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
        return self.laplace_ngram_prob(n_gram_type, word, prev_words)

    def _interpolation_prob(self, word, prev_words:NGram) -> float:
        """lambdas - [unigram, bigram, trigram, ...]"""
        highest_n = 1 + len(prev_words)

        total_prob = 0
        for n in range(highest_n):
            n_gram_type = NGramType(n)
            context = prev_words[-n:] if n > 0 else ()  # last i words
            prob = self.prob_func(n_gram_type, word, context)
            total_prob += self.lambdas[n] * prob

        return total_prob

    def calc_sen_probability(self, sentence:Sentence, n_gram_type:NGramType) -> float:
        #P(w1, w2, ..., wn) = P(wi| wn-t,wn-t)
        log_prob = 0.0

        if n_gram_type == NGramType.INTERPOLATION:
            context = max(NGramType) 
        else:
            context = n_gram_type

        for i in range(context, len(sentence)):
            word = sentence[i]
            prev_words = tuple(sentence[i - context:i])


            if n_gram_type == NGramType.INTERPOLATION:
                word_prob = self._interpolation_prob(word, prev_words)
            else:
                word_prob = self.prob_func(n_gram_type, word, prev_words)

            # Return immediately if probability is zero
            if word_prob <= 0:
                return float('-inf')  
            
            log_prob += math.log(word_prob)

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
        # Start the sentence with a random word if no context is provided
        if not sentence:
            return random.choice(list(self.vocabulary))  

        if n_gram_type == NGramType.INTERPOLATION:
            window = max(NGramType)
        else:
            window = n_gram_type

        prev_words = tuple(sentence[-window:])

        words:list[str] = []
        weights:list[float] = []
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
        start_of_sentence = start_of_sentence.lower()
        processed_start_of_sentence = ["<s>"] + start_of_sentence.split()

        if self.model_type == LanguageModel.UNK:
            processed_start_of_sentence = [w if w in self.vocabulary else "<UNK>" for w in processed_start_of_sentence]

        final_sentence:list[str] = processed_start_of_sentence[:]
        generated_word = ""
        while (len(final_sentence) < max_length) and not generated_word == "</s>": #sentence ends in </s>:
            generated_word = self.generate_next_word(final_sentence, n_gram_type)
            final_sentence.append(generated_word)

        return " ".join(final_sentence[1:-1])
    

    def save_model(self, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{self.model_type}.json")

        readable_model = {
            "model_type": self.model_type,
            "vocabulary": sorted(self.vocabulary),
            "total_tokens": self.total_tokens,
            "lambdas": self.lambdas,
            "ngrams": [
                { " ".join(k): v for k, v in ngram.most_common()}
                for ngram in self.ngrams
            ]
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(readable_model, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        model_type = LanguageModel(data["model_type"])
        model = cls(model_type)

        model.vocabulary = set(data["vocabulary"])
        model.total_tokens = data["total_tokens"]
        model.lambdas = data["lambdas"]

        # Reconstruct n-grams as Counters with tuple keys
        model.ngrams = [
            Counter({tuple(k.split()): v for k, v in ngram_dict.items()})
            for ngram_dict in data["ngrams"]
        ]

        model.prob_func = model.get_prob_function(model_type)

        return model

    def print_memory_usage(self):
        def bytes_to_mb(bytes_val: int) -> float:
            return bytes_val / (1024 * 1024)

        ngrams_memory_bytes = asizeof.asizeof(self.ngrams)
        ngrams_memory_mb = bytes_to_mb(ngrams_memory_bytes)
        print(f"Memory of ngrams (including elements): {ngrams_memory_mb:.2f} MB")

        """# Memory of each ngram (Counter object) and its elements
        for i, ngram in enumerate(self.ngrams):
            ngram_memory_bytes = asizeof.asizeof(ngram)
            ngram_memory_mb = bytes_to_mb(ngram_memory_bytes)
            print(f"Memory of ngram[{i}] (including elements): {ngram_memory_mb:.2f} MB")

        # Memory of vocabulary (set object) and its elements
        vocab_memory_bytes = asizeof.asizeof(self.vocabulary)
        vocab_memory_mb = bytes_to_mb(vocab_memory_bytes)
        print(f"Memory of vocabulary (including elements): {vocab_memory_mb:.2f} MB")"""
