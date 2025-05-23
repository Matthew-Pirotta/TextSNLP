from pre_processing import *
from models import *

import sys
import os
import random
import pandas as pd
import gc
from pympler import asizeof
from collections import Counter

import cProfile, pstats

profile_dir = "./Stats"
models_path = "./Models"

def train_and_eval_all_models(train_sentences, test_sentences, start_sentence, is_training = True) -> tuple:
    columns = [ngram.name for ngram in NGramType]
    index = [model.value for model in LanguageModel]
    perplexity_df = pd.DataFrame(index=index, columns=columns)
    sentence_prob_df = pd.DataFrame(index=index, columns=columns)
    generated_sentence_df = pd.DataFrame(index=index, columns=columns)

    rand_test_sentence = random.choice(test_sentences)

    for model_type in LanguageModel:
        
        if is_training:
                model = Model(model_type)
                model.train(train_sentences)
        else:
                model =  Model.load(f"{models_path}/{model_type}.json")        

        for ngram in NGramType:
            perplexity = model.calc_perplexity(test_sentences, ngram)
            perplexity_df.at[model_type.value, ngram.name] = round(perplexity, 4)

            sen_prob = model.calc_sen_probability(rand_test_sentence, ngram)
            sentence_prob_df.at[model_type.value, ngram.name] = round(sen_prob, 4)

            generated_sentence = model.generate_sentence(start_sentence, NGramType.INTERPOLATION)
            generated_sentence_df.at[model_type.value, ngram.name] = generated_sentence

        
        # append extra info
        sentence_prob_df.at[model_type.value, "Random Test Sentence"] = " ".join(rand_test_sentence)
        generated_sentence_df.at[model_type.value, "start sentence"] = " ".join(start_sentence)
        
        # Explicitly delete the model and run garbage collection
        del model
        gc.collect()
        
    return perplexity_df, sentence_prob_df, generated_sentence_df

#NOTE User input was generated by AI
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

def user_interaction():
    language_model = user_select_model()
    is_training = user_train_or_load()
    start_sentence = user_sentence()
    
    return language_model, is_training, start_sentence

def main_logic_without_user_input(language_model, is_training, start_sentence):    
    corpus = PreProcessing.readSample()

    # Total memory of the corpus
    #corpus_print_memory_usage(corpus)
    
    train_sentences, test_sentences = PreProcessing.train_test_split(corpus, train_ratio=.8)

    
    """model.train(train_sentences)
    model.print_memory_usage()
    generated_sentence = model.generate_sentence(start_sentence, NGramType.BIGRAM)
    print(f"start_sentence: {start_sentence}\nfull generated sentence: {generated_sentence}")"""

    perplexity_df, sentence_prob_df, generated_sentence_df = train_and_eval_all_models(
                                                                train_sentences, test_sentences, start_sentence, is_training)
    print(perplexity_df)

    # Save all DataFrames to their respective files
    dfs = {
        "perplexity_table.csv": perplexity_df,
        "sen_prob_table.csv": sentence_prob_df,
        "generated_sentence_table.csv": generated_sentence_df,
    }

    for file_name, df in dfs.items():
        output_file = os.path.join(models_path, file_name)
        df.to_csv(output_file, index=True)
        print(f"{file_name} saved to {output_file}")


def corpus_print_memory_usage(corpus):
    total_memory_bytes = asizeof.asizeof(corpus)
    total_memory_mb = total_memory_bytes / (1024 * 1024)
    print(f"Total memory used by Corpus: {total_memory_mb:.2f} MB")

def profileTime():
    os.makedirs(profile_dir, exist_ok=True)
    profile_file = os.path.join(profile_dir, "timeResults.cprof")   
    language_model, is_training, start_sentence = user_interaction()

    model = Model(language_model) if is_training else Model.load(f"{models_path}/{language_model}.json")        
    generated_sentence = model.generate_sentence(start_sentence, NGramType.INTERPOLATION)

    print(f"start sentence: {start_sentence}\ngenerated sentence:{generated_sentence}")
    
    with cProfile.Profile() as profile:
        main_logic_without_user_input(language_model, is_training, start_sentence)
        
    stats = pstats.Stats(profile)
    stats.dump_stats(profile_file)
   
    print(f"Profiling results saved to: {profile_file} \n run:'snakeviz Stats/timeResults.cprof'")

if __name__ == '__main__':
    profileTime()

    #Uncomment this and comment above to not profile time
    #language_model, is_training, start_sentence = user_interaction()
    #main_logic_without_user_input(language_model, is_training, start_sentence)

    print("done")