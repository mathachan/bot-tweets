import pandas as pd
from sklearn import preprocessing
import numpy as np

import pickle
import tensorflow as tf

from tensorflow.keras.models import model_from_json

from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub
import json
from tensorflow.keras import layers
import bert



import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
import time
from nltk.stem.porter import PorterStemmer
import numpy as np
nltk.download('stopwords')

from tensorflow.keras.preprocessing.text import Tokenizer


BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
pickle.dump(tokenizer,open('tokenizer.pkl','wb'))
def tokenize_tweet(tweet_text):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweet_text))


json_file = open('model_bilstm201.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("bilstm_weight201.h5")
print("Loaded model from disk")

port_stem = PorterStemmer()

def stemming(content):
    review = re.sub('[^a-zA-Z]',' ',content)
    review = review.lower()
    review = review.split()
    review = [port_stem.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review


#text preprocessing
def cleantext(text):
  x=str(text).lower().replace('\\','').replace('_','')
  tag=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split())
  spcl=tag.replace('[^\w\s]','')
  return spcl


def predictbot(twt):
    print(twt)
    X_data=[]
    twt=cleantext(twt)
    twt=stemming(twt)
    print(twt)
    X_data.append(tokenize_tweet(twt))
    print(X_data)
    test=pad_sequences(X_data, maxlen=68, padding='post')
    ypred=model.predict(test)
    res=np.argmax(ypred[0])
    print("res",res)
    if (res==1):
        return "Not a Bot Tweet"
    else:
        return "Bot Tweet"
   


# predictbot('''#NSFW XXX pics and preview vids from the Den of Debauchery #fetish #tssexychanel http://t.co/NrIkA0hu7s''')
# predictbot('''Barbara Tucker - I Get Lifted (The Bar Dub) http://t.co/MWhgdYd8dy''')
# predictbot('''Good morning and have a wonderful day He preparest a table for me in the presence of mine enemies''')
# predictbot('''i 'm call the only have . world i know you got a ... be be new to great get they have all great day i am have to do n't''')
# predictbot('''the den : man a call me you ... me ... # nsfw delve into the diabolical mind of tssexychanel . xxx adults only blog . you know you wanna  ''')
# predictbot('''@piscesareus : ... # pisces # a @piscesareus : # pisces # pisces @piscesareus : # pisces like i am be know i am am just see i am so''')

# print("-------------------------------------")
# predictbot('''@mulegirl that gif goes right on my default screens''')
# predictbot('''RT @mulegirl: OK, Patricia Arquette brought some Globes level madness.And the Meryl Streep ""right on!"" will be the only gif I use in Slacâ¦''')
# predictbot('''RT @rgay: I love how Meryl jumped up. She is ride or die''')
