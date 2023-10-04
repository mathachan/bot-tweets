import pandas as pd
from sklearn import preprocessing
import numpy as np
# Import train_test_split function
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import  Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
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
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense,Conv1D,MaxPooling1D
from keras.layers import LSTM,Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

nltk.download('stopwords')
#reading dataframe
df = pd.read_csv("dataset2.csv")
print("DATA LOADED\n")
def print_star():
    print('*'*50, '\n')
print(df.head(10))
print_star()
#selecting required columns
df=df[["text","class"]]
df=shuffle(df)
# df=df.head(1000)
print("selecting required columns\n")
print(df.head(10))
print_star()
#Dropping null columns
df=df.dropna( axis=0)
print(df.head(10))
print_star()
# Seperating data and labels



print("Preprocessing Started")


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

print("Preprocessing Completed")
print_star()



df["text"]=df["text"].apply(lambda x:cleantext(x))

data=df["text"]



labels=df["class"]
print(data.head(10))
print(labels.head(10))

BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
pickle.dump(tokenizer,open('tokenizer.pkl','wb'))
def tokenize_tweet(tweet_text):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweet_text))
X_data=[]
Y_data=[]
for i in data:
    X_data.append(tokenize_tweet(i))
for j in labels:
    Y_data.append(j)
# print(X_data[20])
max_length = max([len(s) for s in X_data])
print(max_length)
Xtrain = pad_sequences(X_data, maxlen=max_length, padding='post')
print(Xtrain.shape)
print(len(Y_data))
print("[INFO] Splitting Datas...")
trainX, testX, trainY, testY = train_test_split(Xtrain,Y_data, test_size=0.25, random_state=42)
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)
trainY=np.array(trainY)
# print(trainX)
VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 2

DROPOUT_RATE = 0.2

NB_EPOCHS = 10
# define model
# model = Sequential()
# model.add(Embedding(VOCAB_LENGTH, 128))
# model.add(Conv1D(filters=50, kernel_size=2,padding="valid" ,activation='relu'))
# model.add(Conv1D(filters=50, kernel_size=3,padding="valid" ,activation='relu'))
# model.add(Conv1D(filters=50, kernel_size=4,padding="valid" ,activation='relu'))
# model.add(GlobalMaxPool1D())
# model.add(Dense(units=512, activation="relu"))
# model.add(Dropout(rate=0.1))
# model.add(Dense(units=2,activation="softmax"))
# print(model.summary())

# model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
# history=model.fit(trainX,trainY, epochs=NB_EPOCHS)

model = Sequential()
model.add(Embedding(VOCAB_LENGTH, EMB_DIM, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
filepath="weights_best_cnn1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
callbacks_list = [checkpoint]
model.fit(trainX, trainY, epochs=20, batch_size=256,verbose = 1,callbacks = callbacks_list,validation_data=(testX,testY))

# serialize model to JSON
model_json = model.to_json()
with open("model_lstm20.json", "w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("lstm_weight20.h5")
print("[INFO] Saved model to disk")