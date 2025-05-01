from bs4 import BeautifulSoup
import os
import random
from multiprocessing import Pool

Sentence = list[str]
Sentences = list[Sentence]

random.seed(71)

class PreProcessing:

    #TODO NOTE some sort of filtering? all lower case?
    @staticmethod
    def _extract_sentences(filePath: str) -> Sentences:
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

            if words:
                words.insert(0, "<s>") 
                words.append("</s>")
        
        return sentences
    
    @staticmethod
    def readCorpus() -> Sentences:
        corpusDir:str = "Corpus"
        
        txtFiles = [os.path.normpath(os.path.join(corpusDir, filename))
                    for filename in os.listdir(corpusDir) 
                    if filename.endswith(".vrt")]
        
        with Pool() as pool:
            results = pool.map(PreProcessing._extract_sentences, txtFiles)
        
        corpus = []
        for result in results:
            corpus.extend(result)

        print(f"Total sentences in corpus: {len(corpus)}")
        return corpus
    
    #Reading roughly 1GB of 5.37GB
    @staticmethod
    def readSample(files = None) -> Sentences:
        corpusDir = "Corpus"

        if not files: 
            files = ["malti04.academic.001.vrt", "malti04.administration.001.vrt", "malti04.blogs.001.vrt"]


        corpus = []
        
        for filename in files:
            filePath = os.path.normpath(os.path.join(corpusDir, filename))
            if os.path.isfile(filePath):
                corpus.extend(PreProcessing._extract_sentences(filePath))
  
        print(f"Total sentences in corpus: {len(corpus)}")
        return corpus
    
    @staticmethod
    def train_test_split(sentences:Sentences, train_ratio:float = .8) -> tuple[Sentences,Sentences]:
        random.shuffle(sentences) #Fair randomness

        split_index = int (len(sentences) * train_ratio)

        train_sentences = sentences[:split_index]
        test_sentences = sentences[split_index:]

        print(f"Split corpus: {len(train_sentences)} training sentences, {len(test_sentences)} testing sentences")

        return train_sentences, test_sentences
    
    @staticmethod
    def generate_n_gram(sentence:Sentence,n:int) -> list[tuple[str]]:
        n_grams = []
        upperbound = len(sentence) - n
        for i in range(upperbound+1):
            n_gram = tuple(sentence[i:i+n])
            n_grams.append(n_gram)
        
        return n_grams