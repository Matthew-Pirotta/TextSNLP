# README

## Overview
This project implements a language modeling system capable of training and evaluating unigram, bigram, trigram, and interpolated n-gram models. It also includes functionality for generating random sentences and calculating perplexity for test sentences.

## Saved Models

### Model data
Saved models are stored in the `Models/` directory. Each model is saved as a separate file, with filenames indicating the type of model.

### Overview
The project generates and saves several DataFrames for analysis and evaluation. These include:
1. `perplexity_table.csv`: Contains perplexity scores for test sentences.
2. `sentence_prob_table.csv`: Stores probabilities of test sentences.
3. `generated_sentence.csv`: Contains randomly generated sentences.

## Notes
- Ensure that the `Corpus/` directory contains **Korpus Malti v4.2** for training.