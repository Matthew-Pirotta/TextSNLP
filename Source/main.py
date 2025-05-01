from collections import Counter
from pre_processing import PreProcessing
import time



if __name__ == '__main__':
    start_time = time.time()  # Record the start time
    
    corpus = PreProcessing.readSample()
    train_sentences, test_sentences = PreProcessing.train_test_split(corpus, train_ratio=.8)

    end_time = time.time()  # Record the end time
    print(f"Execution time: {end_time - start_time:.2f} seconds")
