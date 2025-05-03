from pre_processing import *
from models import *

import time
import pandas as pd
import os

models_path = "./Models"

def generate_perplexity_table(train_sentences, test_sentences) -> pd.DataFrame:
    columns = [ngram.name for ngram in NGramType]
    index = [model.value for model in LanguageModel]
    df = pd.DataFrame(index=index, columns=columns)

    for model_type in LanguageModel:
        model = Model(model_type)
        model.train(train_sentences)
        model.save_model(models_path)

        for ngram in NGramType:
            perplexity = model.calc_perplexity(test_sentences, ngram)
            df.at[model_type.value, ngram.name] = round(perplexity, 4)
    
    return df

def user_interaction():
    pass

#TODO Just say chatgpt
def user_select_model() -> LanguageModel:
    while True:
        print("Select a Language Model:")
        for i, model in enumerate(LanguageModel):
            print(f"{i + 1}. {model.value}")
        
        try:
            model_choice = int(input("Enter the number corresponding to your choice: ")) - 1
            if 0 <= model_choice < len(LanguageModel):
                selected_model = list(LanguageModel)[model_choice]
                print(f"Selected Language Model: {selected_model.value}")
                return selected_model
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def user_train_or_load() -> bool:
    while True:
        print("\nDo you want to train a new model or load an existing one?")
        print("1. Train a new model")
        print("2. Load an existing model")
        
        try:
            action_choice = int(input("Enter your choice (1 or 2): "))
            if action_choice == 1:
                return True 
            elif action_choice == 2:
                return False
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number (1 or 2).")

def user_sentence() -> str:
    print("Enter a phrase to generate a sentence:")
    start_sentence = input("Phrase: ")
    return start_sentence



if __name__ == '__main__':

    language_model = user_select_model()
    is_training = user_train_or_load()
    model = Model(language_model) if is_training else Model.load(f"{models_path}/{language_model}")        

    start_sentence = user_sentence() 

    start_time = time.time()  # Record the start time

    corpus = PreProcessing.readSample()
    train_sentences, test_sentences = PreProcessing.train_test_split(corpus, train_ratio=.8)

    model.train(train_sentences)
    
    generated_sentence = model.generate_sentence(start_sentence, NGramType.BIGRAM)
    print(f"start_sentence: {start_sentence}\nfull generated sentence: {generated_sentence}")
    
    

    df = generate_perplexity_table(train_sentences, test_sentences)
    print(df)

    end_time = time.time()  # Record the end time
    print(f"Execution time: {end_time - start_time:.2f} seconds")
