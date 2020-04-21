###############################################################################
## Dataset: Reuters
## w2v parameters: 
## sg: 0 for cbow and 1 for Skip-gram 
## window: The maximum distance between a target word and words around that
## min_count: The minimum count of words to consider when training the model; 
##            words with an occurrence less than this count will be ignored.
## workers: The number of threads to use while training.
## Final Result: Average word embeddings in each document
###############################################################################
import numpy as np
import csv
import nltk
from os.path import isfile
from utils import read_list
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.snowball import PorterStemmer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.corpus import reuters
import pickle
import itertools

###############################################################################
### Fetch the dataset
###############################################################################
category_dict = {'acq':0, 'coffee':1, 'crude':2, 'earn':3, 'gold':4, 'interest':5, 'money-fx':6, 'ship':7, 'sugar':8,
                 'trade':9}
data = []
target = []

nltk.download('reuters')
docs = reuters.fileids()
for doc in docs:
    # Check if the document is only related to 1 class and that class is in category_dict
    if len(reuters.categories(doc)) == 1 and reuters.categories(doc)[0] in category_dict:
        data.append(" ".join(reuters.words(doc))) # Text of the document
        target.append(category_dict[reuters.categories(doc)[0]]) # Index for the class
print("Dataset REUTERS loaded...")
###############################################################################
### Pre-process the dataset
###############################################################################
print("Pre-processing the dataset...")
stemmer = PorterStemmer() # Define the type of stemmer to use
additional_stop_words = []
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
    
    if stemmed_doc == []: # Empty document after pre-processing: to be removed
        id_to_delete.append(i)
    else:
        processed_data.append(stemmed_doc)
data = processed_data
target = np.delete(target, id_to_delete, axis=0)
###############################################################################
### W2V trainer
###############################################################################
window = 10
model_path = "models/reuters_w2v_window" + str(window) + ".model"
if isfile(model_path): # Load if the word2vec model exists
    print("Loading an existing word2vec model trained on the dataset...")
    w2v = Word2Vec.load(model_path)
else: # Otherwise train the word2vec model and save it
    print("Training a word2vec model on the dataset...")
    w2v = Word2Vec(sentences=data, min_count=1, workers=4, sg=1, window=window) # Train a word2vec model on the data
    w2v.save(model_path)
###############################################################################
## Doc2vec by averaging
###############################################################################
print("Building word2vec-based representations of the documents...")

data_w2v = [np.array([w2v[word] for word in doc if word in w2v]) for doc in data]
data_w2v_av = np.array([np.mean([w2v[word] for word in doc if word in w2v], axis=0) for doc in data]) # Average word embeddings in each document
###############################################################################
## Save Results
###############################################################################
with open('data/reuters/data_reuters.txt','wb') as fp:
    pickle.dump(data_w2v, fp)
#
with open('data/reuters/data_reuters_av.txt','wb') as fp:
    pickle.dump(data_w2v_av, fp)
#    
with open('data/reuters/reuters_target.txt','wb') as fp:
    pickle.dump(target, fp)


