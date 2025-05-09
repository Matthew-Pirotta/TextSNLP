import pytest
from collections import Counter
from models import *
from pre_processing import *

def test_generate_unigrams():
    sentence = ["<s>", "the", "cat", "</s>"]
    expected = [("<s>",), ("the",), ("cat",), ("</s>",)]
    assert PreProcessing.generate_n_grams(sentence, 1) == expected

def test_generate_bigrams():
    sentence = ["<s>", "the", "cat", "</s>"]
    expected = [("<s>", "the"), ("the", "cat"), ("cat", "</s>")]
    assert PreProcessing.generate_n_grams(sentence, 2) == expected

def test_vanilla_unigram_prob():
    model = Model(LanguageModel.VANILLA)
    model.ngrams[0] = Counter({("the",): 3, ("cat",): 2})
    model.total_tokens = [5, 0, 0]
    prob = model.vanilla_ngram_prob(NGramType.UNIGRAM, "the", ())
    assert prob == pytest.approx(0.6)

def test_vanilla_bigram_prob():
    model = Model(LanguageModel.VANILLA)
    model.ngrams[1] = Counter({("the", "cat"): 2})
    model.ngrams[0] = Counter({("the",): 4})
    model.total_tokens = [4, 2, 0]
    prob = model.vanilla_ngram_prob(NGramType.BIGRAM, "cat", ("the",))
    
    # count("the cat") / count("the")
    # 2/4 = 1/2
    assert prob == pytest.approx(0.5)

def test_vanilla_trigram_prob():
    model = Model(LanguageModel.VANILLA)

    model.ngrams[2] = Counter({
        ("the", "cat", "sat"): 2,
        ("cat", "sat", "on"): 1
    })
    model.ngrams[1] = Counter({
        ("the", "cat"): 3,
        ("cat", "sat"): 2
    })
    model.ngrams[0] = Counter({
        ("the",): 4,
        ("cat",): 3,
        ("sat",): 2
    })
    model.total_tokens = [9, 5, 3]

    prob = model.vanilla_ngram_prob(NGramType.TRIGRAM, "sat", ("the", "cat"))
    # count("the cat sat") / count("the cat")
    assert prob == pytest.approx(2/3 , abs=1e-6)

def test_vanilla_interpolation_prob():
    model = Model(LanguageModel.VANILLA)

    # Define trigram counts
    model.ngrams[2] = Counter({
        ("the", "cat", "sat"): 2,
        ("cat", "sat", "on"): 1
    })
    model.ngrams[1] = Counter({
        ("the", "cat"): 3,
        ("cat", "sat"): 2
    })
    model.ngrams[0] = Counter({
        ("the",): 4,
        ("cat",): 3,
        ("sat",): 2
    })
    model.total_tokens = [9, 5, 3]

    #sentence = the cat sat
    prob = model._interpolation_prob("sat", ("the", "cat"))
    # (.6 * count("the cat sat") / count("the cat")) + (.3* count("cat sat")/ count("cat")) + (.1* count("sat")/totalTokens)
    # (.6 * 2/3) + (.3 * 2/3) + (.1 * 2/9)
    assert prob == pytest.approx(0.62222222222 , abs=1e-6)

def test_calc_sentence_prob_unigram():
    model = Model(LanguageModel.VANILLA)
    model.ngrams[0] = Counter({("<s>",): 1, ("the",): 1, ("cat",): 1, ("</s>",): 1})
    model.total_tokens = [4, 0, 0]

    sentence = ["<s>", "the", "cat", "</s>"]
    log_prob = model.calc_sen_probability(sentence, NGramType.UNIGRAM)

    #4 * ln(1/4)
    assert log_prob == pytest.approx(-5.54517744448, abs=1e-6)

def test_calc_sentence_prob_bigram():
    model = Model(LanguageModel.VANILLA)
    
    model.ngrams[0] = Counter({("I",): 2, ("walk",): 2, ("<s>",): 1})
    model.ngrams[1] = Counter({
        ("I", "walk"): 2,
        ("<s>", "I"): 1,
        ("walk", "</s>"): 1
    })
    model.total_tokens = [5, 4, 0]

    sentence = ["<s>", "I", "walk", "</s>"]  # Expect "<s> I", "I walk", "walk </s>" lookups

    # ln(1/1) + ln(2/2) + ln(1/2) = -0.69314718056
    log_prob = model.calc_sen_probability(sentence, NGramType.BIGRAM)
    assert log_prob == pytest.approx(-0.69314718056, abs=1e-6)

def test_calc_perplexity_with_vanilla_unigram():
    model = Model(LanguageModel.VANILLA)
    
    train_sentences = [
        ["<s>", "I", "walk", "cats", "</s>"],
        ["<s>", "You", "walk", "dogs", "</s>"]
    ]
    
    model.train(train_sentences)

    # Use one of the training sentences as test data
    test_sentences = [["<s>", "I", "walk", "cats", "</s>"]]

    #log_prob = ln(2/10) + ln(1/10) + ln(2/10) + ln(1/10) + ln(2/10)
    #perplexity = e^(-log_prob/5)
    perplexity = model.calc_perplexity(test_sentences, NGramType.UNIGRAM)

    assert perplexity == pytest.approx(6.59753955386, abs=1e-6)

def test_laplace_ngram_prob():
    model = Model(LanguageModel.LAPLACE)
    model.vocabulary = {"the", "cat", "</s>"}
    
    model.ngrams[0] = Counter({("the",): 2})
    model.ngrams[1] = Counter({("the", "cat"): 1})
    model.total_tokens = [2, 1, 0]

    prob = model.laplace_ngram_prob(NGramType.BIGRAM, "cat", ("the",))
    expected = (1 + 1) / (2 + 3)  # (count + 1) / (prev_count + vocab_size)
    assert prob == pytest.approx(expected, abs=1e-6)

def test_unk_ngram_prob():
    model = Model(LanguageModel.UNK)

    train_sentences = [
        ["<s>", "the", "cat", "sat", "</s>"],
        ["<s>", "the", "dog", "barked", "</s>"]
    ]
    model.train(train_sentences)

    # Check that rare words are replaced with <UNK>
    assert "<UNK>" in model.vocabulary
    assert "barked" not in model.vocabulary 

    sentence = ["<s>", "the", "cat", "barked", "</s>"]
    log_prob = model.calc_sen_probability(sentence, NGramType.UNIGRAM)

    assert log_prob <= 0

    # Check that <UNK> is used in place of "barked"
    unk_prob = model.unk_ngram_prob(NGramType.UNIGRAM, "barked", ())
    assert unk_prob > 0 