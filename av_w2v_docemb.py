#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:13:12 2019
In this program first we choose dataset from our text collection:
    - 20news
    - Agnews
    - Dbpedia
    - Reuters
    - Yahoo
Final goal is achieving document vector through averaging word vectors.
Some tips!
    - w2v parameters: 
        * sg: 0 for cbow and 1 for Skip-gram 
        * window: The maximum distance between a target word and words around that
        * min_count: The minimum count of words to consider when training the model; 
          words with an occurrence less than this count will be ignored.
        * workers: The number of threads to use while training.
    - Final Result: Average word embeddings in each document
@author: khosravm
"""
###############################################################################
## Importing requirements
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
import csv
import nltk

###############################################################################
### Choose and load dataset
###############################################################################
"""Inorder to be applicable for all of 5 datasets from our text collection 
   we define choicetxtcoll() to select dataset by user."""

def choicetxtcoll():
    print('''Enter your choice for dataset:
        1 20news
        2 Agnews
        3 Dbpedia
        4 Reuters
        5 Yahoo''')
    y = input()
    x = int(y)
    target = []
    data = []
    
    if x == 1:
      _20news = fetch_20newsgroups(subset="all")
      print("Dataset 20NEWS loaded...")
      data = _20news.data
      target = _20news.target
    elif x == 2:
      with open('data/agnewstrain.csv', 'r', encoding='utf-8') as f:
          csv_file = csv.reader(f, delimiter=',')
          for row in csv_file:
              target.append(int(row[0])) # Class index
              data.append(row[2].encode('utf-8', 'ignore')) # Text description (ignore the entity name)
      data = np.asarray(data)
      target = np.asarray(target)
      target = target - 1 # Labels starting from 0
      print("Dataset AGNEWS loaded...")
    elif x == 3:
      with open('data/dbpediatest.csv', 'r', encoding='utf-8') as f:
          csv_file = csv.reader(f, delimiter=',')
          for row in csv_file:
              target.append(int(row[0])) # Class index
              data.append(row[2].encode('utf-8', 'ignore')) # Text description (ignore the entity name)
      data = np.asarray(data)
      target = np.asarray(target)
      target = target - 1 # Labels starting from 0
      print("Dataset DBPEDIA loaded...")  
    elif x == 4:
      from nltk.corpus import reuters  
      category_dict = {'acq':0, 'coffee':1, 'crude':2, 'earn':3, 'gold':4, 'interest':5, 'money-fx':6, 'ship':7, 'sugar':8,
                     'trade':9}
      
      nltk.download('reuters')
      docs = reuters.fileids()
      for doc in docs:
          # Check if the document is only related to 1 class and that class is in category_dict
          if len(reuters.categories(doc)) == 1 and reuters.categories(doc)[0] in category_dict:
              data.append(" ".join(reuters.words(doc))) # Text of the document
              target.append(category_dict[reuters.categories(doc)[0]]) # Index for the class
      print("Dataset REUTERS loaded...")  
    elif x == 5:
        with open('data/yahootrain.csv', 'r', encoding='utf-8') as f:
            csv_file = csv.reader(f, delimiter=',')
            for row in csv_file:
                target.append(int(row[0])) # Class index
                data.append((row[1] + " ").encode('utf-8', 'ignore') + # Question title
                            (row[2] + " ").encode('utf-8', 'ignore') + # Full question
                             row[3].encode('utf-8', 'ignore')) # Best answer
        data = np.asarray(data)
        target = np.asarray(target)
        target = target - 1 # Labels starting from 0
        print("Dataset YAHOO loaded...")
    else:
      print("Wrong choice! Rerun with a true choice.")
      print("****************************************")
      choice()
    return data,target,y

data,target,y = choicetxtcoll()       

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

window = 50    #10
model_path = "models/dataset" + str(y) + "_w2v_window" + str(window) + ".model"
if isfile(model_path): # Load if the word2vec model exists
    print("Loading an existing word2vec model trained on the dataset...")
    w2v = Word2Vec.load(model_path)
else: # Otherwise train the word2vec model and save it
    print("Training a word2vec model on the dataset...")
    # Train a word2vec model on the data (sg = 0 for cbow and 1 for skip-gram)
    # Default vector size is 100 and we do not change it here.
    w2v = Word2Vec(sentences=data, min_count=1, workers=4, sg=1, window=window) 
    w2v.save(model_path)
    
  
###############################################################################
## Doc2vec by averaging
###############################################################################
print("Building word2vec-based representations of the documents...")

#data_w2v = [[w2v[word] for word in doc if word in w2v] for doc in data]
data_w2v = [np.array([w2v[word] for word in doc if word in w2v]) for doc in data]
#data_w2v_av = [np.mean([w2v[word] for word in doc if word in w2v], axis=0) for doc in data]
data_w2v_av = np.array([np.mean([w2v[word] for word in doc if word in w2v], axis=0) for doc in data]) # Average word embeddings in each document 

###############################################################################
## Save Results
###############################################################################
with open('data/results/dataset.txt','wb') as fp:
    pickle.dump(data_w2v, fp)

with open('data/results/dataset_av.txt','wb') as fp:
    pickle.dump(data_w2v_av, fp)
    
with open('data/results/finaltarget.txt','wb') as fp:
    pickle.dump(target, fp)

