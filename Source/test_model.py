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
    assert prob == pytest.approx(0.5)

def test_calc_sentence_prob_unigram():
    model = Model("test")
    model.ngrams[0] = Counter({("<s>",): 1, ("the",): 1, ("cat",): 1, ("</s>",): 1})
    model.total_tokens = [4, 0, 0]

    sentence = ["<s>", "the", "cat", "</s>"]
    log_prob = model.calc_sen_probability(sentence, NGramType.UNIGRAM, model.vanilla_ngram_prob)
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
    log_prob = model.calc_sen_probability(sentence, NGramType.BIGRAM, model.vanilla_ngram_prob)
    
    assert isinstance(log_prob, float)
    assert log_prob < 0  # Should be negative since some probabilities < 1

#TODO wrong for now need to see stuff
"""def test_vanilla_interpolation_prob():
    model = Model("test")

    model.ngrams[0] = Counter({("<s>",): 2,("the",): 2,("cat",): 1,("</s>",): 1 })
    model.ngrams[1] = Counter({("<s>", "the"): 2,("the", "cat"): 1,("cat", "</s>"): 1})
    model.total_tokens = [6, 4, 0]

    sentence = ["<s>", "the", "cat", "</s>"]

    log_prob = model.calc_n_gram_sent_prob(sentence, NGramType.BIGRAM, model.vanilla_ngram_prob)

    expected_log_prob = -0.6931  # from log(0.5)
    assert log_prob == pytest.approx(expected_log_prob, abs=1e-4)"""

def test_calc_perplexity_with_vanilla_unigram():
    model = Model("perplexity_vanilla")
    
    train_sentences = [
        ["<s>", "I", "love", "cats", "</s>"],
        ["<s>", "You", "love", "dogs", "</s>"]
    ]
    
    model.train(train_sentences)

    # Use one of the training sentences as test data
    test_sentences = [["<s>", "I", "love", "cats", "</s>"]]

    perplexity = model.calc_perplexity(test_sentences, NGramType.UNIGRAM, model.vanilla_ngram_prob)

    assert perplexity == pytest.approx(6.6, abs=1e-2)


def test_laplace_ngram_prob():
    model = Model("test")
    model.vocabulary = {"the", "cat", "</s>"}
    
    model.ngrams[0] = Counter({("the",): 2})
    model.ngrams[1] = Counter({("the", "cat"): 1})
    model.total_tokens = [2, 1, 0]

    prob = model.laplace_ngram_prob(NGramType.BIGRAM, "cat", ("the",))
    expected = (1 + 1) / (2 + 3)  # (count + 1) / (prev_count + vocab_size)
    assert prob == pytest.approx(expected, abs=1e-6)


# ---------- Edge Case ----------

def test_empty_sentence_prob():
    model = Model("empty")
    model.train([["<s>", "</s>"]])
    prob = model.calc_sen_probability(["<s>", "</s>"], NGramType.UNIGRAM, model.vanilla_ngram_prob)
    assert prob <= 0
