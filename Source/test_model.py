import pytest
from collections import Counter
from models import Model, NGramType
from pre_processing import PreProcessing

# Mock types for convenience
Sentence = list[str]
NGram = tuple[str]

# ---------- Unit Tests ----------

def test_generate_unigrams():
    sentence = ["<s>", "the", "cat", "</s>"]
    expected = [("<s>",), ("the",), ("cat",), ("</s>",)]
    assert PreProcessing.generate_n_grams(sentence, 1) == expected

def test_generate_bigrams():
    sentence = ["<s>", "the", "cat", "</s>"]
    expected = [("<s>", "the"), ("the", "cat"), ("cat", "</s>")]
    assert PreProcessing.generate_n_grams(sentence, 2) == expected

def test_vanilla_unigram_prob():
    model = Model("test")
    model.ngrams[0] = Counter({("the",): 3, ("cat",): 2})
    model.total_tokens = [5, 0, 0]
    assert pytest.approx(model.vanilla_ngram_prob(NGramType.UNIGRAM, "the", ())) == 0.6

def test_vanilla_bigram_prob():
    model = Model("test")
    model.ngrams[1] = Counter({("the", "cat"): 2})
    model.ngrams[0] = Counter({("the",): 4})
    model.total_tokens = [4, 2, 0]
    prob = model.vanilla_ngram_prob(NGramType.BIGRAM, "cat", ("the",))
    assert pytest.approx(prob) == 0.5

def test_calc_sentence_prob_unigram():
    model = Model("test")
    model.ngrams[0] = Counter({("<s>",): 1, ("the",): 1, ("cat",): 1, ("</s>",): 1})
    model.total_tokens = [4, 0, 0]

    sentence = ["<s>", "the", "cat", "</s>"]
    log_prob = model.calc_sentence_prob(sentence, NGramType.UNIGRAM, model.vanilla_ngram_prob)
    assert log_prob < 0  # Should be negative log-probability

def test_calc_sentence_prob_bigram():
    model = Model("bigram_test")
    
    # Manually populate bigram and unigram counts
    model.ngrams[0] = Counter({("I",): 2, ("love",): 2})
    model.ngrams[1] = Counter({("I", "love"): 2})
    model.total_tokens = [4, 2, 0]  # total unigrams: 4; total bigrams: 2

    sentence = ["<s>", "I", "love", "</s>"]  # Expect "<s> I", "I love", "love </s>" lookups
    model.ngrams[0][("<s>",)] = 1
    model.ngrams[1][("<s>", "I")] = 1
    model.ngrams[1][("love", "</s>")] = 1
    model.total_tokens[0] += 1
    model.total_tokens[1] += 2

    # Run bigram probability calculation
    log_prob = model.calc_sentence_prob(sentence, NGramType.BIGRAM, model.vanilla_ngram_prob)
    
    assert isinstance(log_prob, float)
    assert log_prob < 0  # Should be negative since some probabilities < 1

# ---------- Edge Case ----------

def test_empty_sentence_prob():
    model = Model("empty")
    model.train([["<s>", "</s>"]])
    prob = model.calc_sentence_prob(["<s>", "</s>"], NGramType.UNIGRAM, model.vanilla_ngram_prob)
    assert prob <= 0
