from collections import Counter
import os
from multiprocessing import Pool, cpu_count
from bs4 import BeautifulSoup

def readFile(filePath:str)-> Counter:
    wordOccurrences = Counter()
    #NOTE do not have 'c' and 'y'
    acceptedCharacters = {'a', 'b', 'ċ', 'd', 'e', 'f', 'g', 'ġ', 'h', 'ħ', 'i', 'j', 
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z', 'ż', "'", '-'}
    with open(filePath, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip().split()
            if not words: continue #skip empty lines

            word = words[0] #only interested in first column
            word = word.lower()
            
            if all(char in acceptedCharacters for char in word):
               # print(word)
                wordOccurrences[word] += 1                
    print(f"loaded document {filePath} with {wordOccurrences.total()} words")
    return wordOccurrences


def extract_sentences_first_column(vrt_text):
    soup = BeautifulSoup(vrt_text, "xml")
    sentences = []
    
    for s_tag in soup.find_all("s"):
        words = []
        for line in s_tag.get_text().strip().split("\n"):
            if line.strip():
                parts = line.strip().split("\t")
                if len(parts) > 0:
                    words.append(parts[0])
        sentences.append(words)
    
    return sentences



def readCorpus() -> Counter:
    corpusDir = "Corpus"
    
    txtFiles = [os.path.join(corpusDir, filename) 
                for filename in os.listdir(corpusDir) 
                if filename.endswith(".txt")]
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(readFile, txtFiles)
    
    wordOccurrences = Counter()
    for result in results:
        wordOccurrences.update(result)
    
    print(f"Total words in corpus: {wordOccurrences.total()}")
    return wordOccurrences

#Reading roughly 1GB of 5.37GB
def readSample() -> Counter:
    corpusDir = "Corpus"
    
    files = ["malti03.parl.4.txt", "malti03.law.txt", "malti03.opinion.2.txt", "malti03.news.1.txt"]
    wordOccurrences = Counter()
    
    for filename in files:
        filePath = os.path.join(corpusDir, filename)
        if os.path.isfile(filePath):
            subOccurrence = readFile(filePath)
            wordOccurrences.update(subOccurrence)
    print(f"num words in sample: {wordOccurrences.total()}")
    return wordOccurrences

def generate_n_gram(sentence:list[str],n:int) -> list[list[str]]:
    ngrams = []
    upperbound = len(sentence) - n
    for i in range(upperbound+1):
        ngram = sentence[i:i+n]
        ngrams.append(ngram)
    
    return ngrams


if __name__ == '__main__':
    with open("Corpus/malti04.academic.001.vrt", "r", encoding="utf-8") as file:
        vrt_text = file.read()

        sentences = extract_sentences_first_column(vrt_text)

        # Show first 2 sentences
        for i, sent in enumerate(sentences[:2]):
            print(f"Sentence {i+1}: {sent}")

        print("--------------")
        print(generate_n_gram(sentences[1], 1))
        print("--------------")
        print(generate_n_gram(sentences[1], 2))
        print("--------------")
        print(generate_n_gram(sentences[1], 3))