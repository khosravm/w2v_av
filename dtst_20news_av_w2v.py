###############################################################################
## Dataset: 20news
## w2v parameters: 
## sg: 0 for cbow and 1 for Skip-gram 
## window: The maximum distance between a target word and words around that
## min_count: The minimum count of words to consider when training the model; 
##            words with an occurrence less than this count will be ignored.
## workers: The number of threads to use while training.
## Final Result: Average word embeddings in each document
###############################################################################

from utils import read_list
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from os.path import isfile
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.snowball import PorterStemmer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import itertools

###############################################################################
### Fetch the dataset
###############################################################################
_20news = fetch_20newsgroups(subset="all")
print("Dataset 20NEWS loaded...")
data = _20news.data
target = _20news.target
###############################################################################
# Pre-process the dataset
###############################################################################
print("Pre-processing the dataset...")
stemmer = PorterStemmer() # Define the type of stemmer to use
additional_stop_words = ['edu', 'com', 'gov', 'ca', 'mit', 'uk', 'subject', 'lines', 'organization', 'writes', 'msg',
                         'article', 'university', 'does', 'posting', 'thanks', 'don', 'know', 'help', 'use', 'copy']
stop_words = ENGLISH_STOP_WORDS.union(additional_stop_words)
stop_words = set([stemmer.stem(word) for word in stop_words]) # Stem the stop words for larger detection
processed_data = []
id_to_delete = []
for i, doc in enumerate(data):
    tokenized_doc = list(simple_preprocess(doc, deacc=True, min_len=2))
    stemmed_doc = []
    for word in tokenized_doc:
        stemmed_word = stemmer.stem(word)
        if stemmed_word not in stop_words:
            stemmed_doc.append(stemmed_word)
    #[stemmer.stem(word) for word in tokenized_doc if word not in stop_words]
    if stemmed_doc == []: # Empty document after pre-processing: to be removed
        id_to_delete.append(i)
    else:
        processed_data.append(stemmed_doc)
data = processed_data
target = np.delete(target, id_to_delete, axis=0)
###############################################################################
## Word2vec Trainer
###############################################################################

window = 50
model_path = "models/20news_w2v_window" + str(window) + ".model"
if isfile(model_path): # Load if the word2vec model exists
    print("Loading an existing word2vec model trained on the dataset...")
    w2v = Word2Vec.load(model_path)
else: # Otherwise train the word2vec model and save it
    print("Training a word2vec model on the dataset...")
    # Train a word2vec model on the data (sg = 0 for cbow and 1 for skip-gram)
    w2v = Word2Vec(sentences=data, min_count=1, workers=4, sg=1, window=window) 
    w2v.save(model_path)

###############################################################################
## Doc2vec by averaging
###############################################################################
print("Building word2vec-based representations of the documents...")
data_w2v = [[w2v[word] for word in doc if word in w2v] for doc in data]
# Average word embeddings in each document
data_w2v_av = [np.mean([w2v[word] for word in doc if word in w2v], axis=0) for doc in data] 
###############################################################################
## Save Results
###############################################################################
with open('data/20news/data_20news.txt','wb') as fp:
    pickle.dump(data_w2v, fp)

with open('data/20news/data_20news_av.txt','wb') as fp:
    pickle.dump(data_w2v_av, fp)
    
with open('data/20news/20news_target.txt','wb') as fp:
    pickle.dump(target, fp)

