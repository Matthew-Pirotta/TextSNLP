from collections import Counter
import os
from multiprocessing import Pool, cpu_count
from bs4 import BeautifulSoup

#TODO NOTE some sort of filtering? all lower case?
def readFile(filePath:str, n:int) -> Counter:
    n_gram_occurrences = Counter()

    sentences = extract_sentences(filePath)

    for sentence in sentences:
        for n_gram in generate_n_gram(sentence, n):
            n_gram_occurrences[tuple(n_gram)] += 1

    print(f"loaded document {filePath} with {n_gram_occurrences.total()} {n}-grams")
    return n_gram_occurrences

def extract_sentences(filePath: str) -> list[list[str]]:
    with open(filePath, "r", encoding="utf-8") as file:
        vrt_text = file.read()

    soup = BeautifulSoup(vrt_text, "lxml-xml")
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



def readCorpus(n:int) -> Counter:
    corpusDir = "Corpus"
    
    txtFiles = [os.path.join(corpusDir, filename) 
                for filename in os.listdir(corpusDir) 
                if filename.endswith(".vrt")]
    
    args = [(filePath, n) for filePath in txtFiles]
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(readFile, args)
    
    n_gram_occurrences = Counter()
    for result in results:
        n_gram_occurrences.update(result)
    
    print(f"Total words in corpus: {n_gram_occurrences.total()}")
    return n_gram_occurrences

#Reading roughly 1GB of 5.37GB
def readSample(n:int) -> Counter:
    corpusDir = "Corpus"
    
    files = ["malti04.academic.001.vrt", "malti04.administration.001.vrt", "malti04.blogs.001.vrt"]
    n_gram_occurrences = Counter()
    
    for filename in files:
        filePath = os.path.join(corpusDir, filename)
        if os.path.isfile(filePath):
            subOccurrence = readFile(filePath,n)
            n_gram_occurrences.update(subOccurrence)
    print(f"num words in sample: {n_gram_occurrences.total()}")
    return n_gram_occurrences

def generate_n_gram(sentence:list[str],n:int) -> list[tuple[str]]:
    n_grams = []
    upperbound = len(sentence) - n
    for i in range(upperbound+1):
        n_gram = tuple(sentence[i:i+n])
        n_grams.append(n_gram)
    
    return n_grams


if __name__ == '__main__':
    n_gram_occurrences = readCorpus(2)
    #print(n_gram_occurrences)
