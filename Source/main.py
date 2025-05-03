from pre_processing import *
from models import *

import time
import pandas as pd


if __name__ == '__main__':
    start_time = time.time()  # Record the start time
    
    corpus = PreProcessing.readSample()
    train_sentences, test_sentences = PreProcessing.train_test_split(corpus, train_ratio=.8)

    model = Model("Base", LanguageModel.VANILLA)
    model.train(train_sentences)

    columns = [ngram.name for ngram in NGramType]
    index = [model.value for model in LanguageModel]
    df = pd.DataFrame(index=index, columns=columns)

    for model_type in LanguageModel:
        #NOTE This assumes that unk is the last model
        #Since all the other models are the same, can reuse, but need to reprocess that for UNK
        #This is an optimization that skips unnecessary retraining 
        if model_type == LanguageModel.UNK:
            model = Model("UNK", LanguageModel.UNK)
            model.train(train_sentences)
        else:
            processed_test = test_sentences
        
        for ngram in NGramType:
            perplexity = model.calc_perplexity(processed_test, ngram)
            df.at[model_type.value, ngram.name] = round(perplexity, 4)

    print(df)

    text = model.generate_sentence("Jien jisimni", NGramType.BIGRAM)
    print(f"text- {text}")

    end_time = time.time()  # Record the end time
    print(f"Execution time: {end_time - start_time:.2f} seconds")