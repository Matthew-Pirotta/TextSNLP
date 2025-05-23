from bs4 import BeautifulSoup
import os
import random
from multiprocessing import Pool

Sentence = list[str]
Sentences = list[Sentence]
NGram = tuple[str, ...]
NGrams = list[NGram]

__all__ = ["Sentence", "Sentences", "NGram", "NGrams", "PreProcessing"]

random.seed(71)

class PreProcessing:
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
                        words.append(parts[0].lower())
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
    
    @staticmethod
    def get_largest_files_by_category(corpus_dir, num_files=3):
        category_files = {}

        for filename in os.listdir(corpus_dir):
            if filename.endswith(".vrt"):
                #eg, "malti04.administration.085.vrt"
                parts = filename.split(".")
                if len(parts) > 1:
                    category = parts[1]
                    file_path = os.path.join(corpus_dir, filename)
                    file_size = os.path.getsize(file_path)

                    # Add the file to the category
                    if category not in category_files:
                        category_files[category] = []
                    category_files[category].append((filename, file_size))

        #print(category_files)
        largest_files = []
        for category, files in category_files.items():
            files.sort(key=lambda x: x[1], reverse=True)
            largest_files.extend([file[0] for file in files[:num_files]])

        return largest_files
    
    #Reading roughly 3GB
    @staticmethod
    def readSample(files = None) -> Sentences:
        corpusDir = "Corpus"

        if not files: 
            #Biggest files ordered - 
            #files = ["malti04.administration.085.vrt", "malti04.web.vrt", "malti04.parliament.008.vrt", "malti04.parliament.009.vrt"]

            #Biggest 4 files from each section
            files = PreProcessing.get_largest_files_by_category(corpusDir, 10)

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

        def bytes_to_mb(bytes_val: int) -> float:
            return bytes_val / (1024 * 1024)

        print(f"Split corpus: {len(train_sentences)} training sentences, {len(test_sentences)} testing sentences")

        return train_sentences, test_sentences
    
    @staticmethod
    def generate_n_grams(sentence:Sentence, n:int) -> NGrams:
        n_grams = []
        upperbound = len(sentence) - n
        for i in range(upperbound+1):
            n_gram = tuple(sentence[i:i+n])
            n_grams.append(n_gram)
        
        return n_grams