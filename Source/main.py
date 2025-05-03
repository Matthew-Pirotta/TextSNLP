from pre_processing import *
from models import *

import time
import pandas as pd

def generate_perplexity_table(train_sentences, test_sentences) -> pd.DataFrame:
    columns = [ngram.name for ngram in NGramType]
    index = [model.value for model in LanguageModel]
    df = pd.DataFrame(index=index, columns=columns)

    for model_type in LanguageModel:
        model = Model("Base", model_type)
        model.train(train_sentences)

        for ngram in NGramType:
            perplexity = model.calc_perplexity(test_sentences, ngram)
            df.at[model_type.value, ngram.name] = round(perplexity, 4)
    
    return df

if __name__ == '__main__':
    start_time = time.time()  # Record the start time
    
    corpus = PreProcessing.readSample()
    train_sentences, test_sentences = PreProcessing.train_test_split(corpus, train_ratio=.8)


    model = Model("idk", LanguageModel.VANILLA)
    start_sentence = "Jien inhobb qtates"
    generated_sentence = model.generate_sentence(start_sentence, NGramType.BIGRAM)
    print(f"start_sentence: {start_sentence}\nfull generated sentence: {generated_sentence}")

    #df = generate_perplexity_table(train_sentences, test_sentences)
    #print(df)

    end_time = time.time()  # Record the end time
    print(f"Execution time: {end_time - start_time:.2f} seconds")
