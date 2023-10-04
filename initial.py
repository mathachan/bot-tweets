
import pandas as pd
#import spacy, and spacy model
# spacy is used to work on text
import spacy
nlp = spacy.load('en_core_web_sm')
import re
#import other libraries
import numpy as np
import random
import sys
import os
import time
import codecs
import collections
from six.moves import cPickle


seq_length = 30 
df=pd.read_csv('spam2.csv')
save_dir = 'save' # directory to store trained NN models
vocab_file = os.path.join(save_dir, "words_vocab.pkl")
sequences_step = 1 #step to create sequences

def create_wordlist(doc):
    wl = []
    for word in doc:
        if word.text not in ("\n","\n\n",'\u2009','\xa0'):
            wl.append(word.text.lower())
    return wl


wordlist = []

datas=df['text']
datas=datas.head(100)
print(datas.head(10))
for dt in datas:
    
    
    try: 
        result = re.sub(r'http\S+', '', dt)
        print(result)
        #create sentences
        doc = nlp(result)
        wl = create_wordlist(doc)
        wordlist = wordlist + wl
    except:
        pass

# print(wordlist)



# count the number of words
word_counts = collections.Counter(wordlist)

# Mapping from index to word : that's the vocabulary
vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))

# Mapping from word to index
vocab = {x: i for i, x in enumerate(vocabulary_inv)}
words = [x[0] for x in word_counts.most_common()]

#size of the vocabulary
vocab_size = len(words)
print("vocab size: ", vocab_size)

#save the words and vocabulary
with open(os.path.join(vocab_file), 'wb') as f:
    cPickle.dump((words, vocab, vocabulary_inv), f)


#create sequences
sequences = []
next_words = []
for i in range(0, len(wordlist) - seq_length, sequences_step):
    sequences.append(wordlist[i: i + seq_length])
    next_words.append(wordlist[i + seq_length])

print('nb sequences:', len(sequences))
