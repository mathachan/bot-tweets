

######### Train BiLSTM language model #####################

from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.metrics import categorical_accuracy
import pandas as pd
from keras.models import Sequential, Model

import spacy
nlp = spacy.load('en_core_web_sm')
import re
#import other libraries
import numpy as np

import os

import collections
from six.moves import cPickle


seq_longueur = 30 
Les_df=pd.read_csv('spam2.csv')
Sv_dir = 'save'
vocab_sq = os.path.join(Sv_dir, "Mots_vocab.pkl")
Dossier  = 1 

def w_list_create(doc):
    w01 = []
    for Mot in doc:
        if Mot.text not in ("\n","\n\n",'\u2009','\xa0'):
            w01.append(Mot.text.lower())
    return w01


Motlist = []
donnees=Les_df['text']
donnees=donnees.head(100)
print(donnees.head(10))
for dt in donnees:
    try: 
        res_fr = re.sub(r'http\S+', '', dt)
        print(res_fr)
        #create phrases
        doc = nlp(res_fr)
        w01 = w_list_create(doc)
        Motlist = Motlist + w01
    except:
        pass




Mot_counts = collections.Counter(Motlist)


vocabulary_inv = [x[0] for x in Mot_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))

vocab = {x: i for i, x in enumerate(vocabulary_inv)}
Mots = [x[0] for x in Mot_counts.most_common()]

vosize = len(Mots)
print("vocab size: ", vosize)

with open(os.path.join(vocab_sq), 'wb') as f:
    cPickle.dump((Mots, vocab, vocabulary_inv), f)


sqnces = []
next_Mots = []
for i in range(0, len(Motlist) - seq_longueur, Dossier ):
    sqnces.append(Motlist[i: i + seq_longueur])
    next_Mots.append(Motlist[i + seq_longueur])

print('nb sqnces:', len(sqnces))



X_dnum= np.zeros((len(sqnces), seq_longueur, vosize), dtype=np.bool)
y_dnum = np.zeros((len(sqnces), vosize), dtype=np.bool)
for i, phrase in enumerate(sqnces):
    for t, Mot in enumerate(phrase):
        X_dnum[i, t, vocab[Mot]] = 1
    y_dnum[i, vocab[next_Mots[i]]] = 1

rnn_size = 256 
seq_longueur = 30 
learning_rate = 0.001 
def bi_di_lmodel(seq_longueur, vosize):
    print('Build LSTM model.')
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_size, activation="relu"),input_shape=(seq_longueur, vosize)))
    model.add(Dropout(0.6))
    model.add(Dense(vosize))
    model.add(Activation('softmax'))
    
    optimizer = Adam(lr=learning_rate)
    callbacks=[EarlyStopping(patience=2, monitor='val_loss')]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    print("model built!")
    return model



md = bi_di_lmodel(seq_longueur, vosize)
print(md.summary())


lot_size = 32 
num_epochs = 10

rappels=[EarlyStopping(patience=4, monitor='val_loss'),
           ModelCheckpoint(filepath=Sv_dir + "/" + 'my_model_gen_phrases1.{epoch:02d}-{val_loss:.2f}.hLes_df5',\
                           monitor='val_loss', verbose=0, mode='auto', period=2)]
histoire = md.fit(X_dnum, y_dnum,
                 batch_size=lot_size,
                 shuffle=True,
                 epochs=num_epochs,
                 callbacks=rappels,
                 validation_split=0.1)

md.save(Sv_dir + "/" + 'my_model_produire_phrases.h5')

########### Test Bi LSTM language model ########



import os
import numpy as np
import pandas as pd
  

from six.moves import cPickle
Sv_dir = 'saventree'

print("loading vocabulary...")
vocab_sq = os.path.join(Sv_dir, "Mots_vocab.pkl")

with open(os.path.join(Sv_dir, 'Mots_vocab.pkl'), 'rb') as f:
        Mots, vocab, vocabulary_inv = cPickle.load(f)

vosize = len(Mots)

from keras.models import load_model
print("loading model...")
model = load_model(Sv_dir + "/" + 'my_model_gen_phrases.06-5.83.hLes_df5')


def sample(preds_, temperature=1.0):
    
    preds_ = np.asarray(preds_).astype('float64')
    preds_ = np.log(preds_) / temperature
    exp_preds_ = np.exp(preds_)
    preds_ = exp_preds_ / np.sum(exp_preds_)
    probas = np.random.multinomial(1, preds_, 1)
    return np.argmax(probas)

seq_longueur = 30 
Mots_number = 30 
def produire(sen):
    planter = sen 
    produired = ''
    phrase = []

    for i in range (seq_longueur):
        phrase.append("a")

    seed = planter.split()

    for i in range(len(seed)):
        phrase[seq_longueur-i-1]=seed[len(seed)-i-1]

    produired += ' '.join(phrase)

    for i in range(Mots_number):
        x = np.zeros((1, seq_longueur, vosize))
        for t, Mot in enumerate(phrase):
            x[0, t, vocab[Mot]] = 1.

        preds_ = model.predict(x, verbose=0)[0]
        next_indice = sample(preds_, 0.33)
        next_Mot = vocabulary_inv[next_indice]

      
        produired += " " + next_Mot
        phrase = phrase[1:] + [next_Mot]
    senval=""
    cnt=0
    for i in range(len(produired)-2):
        
        if(produired[i]=='a' and produired[i+2]=='a'):
            cnt=i+1
        else:
            pass
    print(cnt)
    senval=produired[cnt+3:]
    
    print(senval)

    return senval
slist=[]
for i in Mots[:300]:
    try:
        senval=produire(i)
        slist.append(senval)
    except:
        pass


Les_df = pd.DataFrame(slist,columns =['text'])
print(Les_df)
Les_df.to_csv('out.csv',index=False)


################## Train CNN LSTM Model ###################


import nltk
import re

from nltk.stem.porter import PorterStemmer
import numpy as np
from nltk.corpus import stopwords
from tensorflow.keras.layers import Conv1D,Dense,Embedding

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub
import bert
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense,Conv1D,MaxPooling1D,LSTM
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd

nltk.download('stopMots')
Les_df = pd.read_csv("donneeset1.csv")
print("DATA LOADED\n")
def print_star():
    print('*'*50, '\n')
print(Les_df.head(10))
print_star()
Les_df=Les_df[["text","class"]]
Les_df=shuffle(Les_df)
print("selecting required columns\n")
print(Les_df.head(10))
print_star()
Les_df=Les_df.dropna( axis=0)
print(Les_df.head(10))
print_star()



print("Preprocessing Started")


ecorcher = PorterStemmer()

def stemming(content):
    Mot_content = re.sub('[^a-zA-Z]',' ',content)
    Mot_content = Mot_content.lower()
    Mot_content = Mot_content.split()
    Mot_content = [ecorcher.stem(Mot) for Mot in Mot_content if not Mot in stopwords.words('english')]
    Mot_content = ' '.join(Mot_content)
    return Mot_content


def nettoyer_text(text):
  x=str(text).lower().replace('\\','').replace('_','')
  tag=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split())
  spcl=tag.replace('[^\w\s]','')
  return spcl

print("Preprocessing Completed")
print_star()



Les_df["text"]=Les_df["text"].apply(lambda x:nettoyer_text(x))
Les_df["text"]=Les_df["text"].apply(lambda x:stemming(x))

data=Les_df["text"]


Bert_jeton = bert.bert_tokenization.Fulljetonizer
labels=Les_df["class"]
print(data.head(10))
print(labels.head(10))


bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_sq.asset_path.numpy()
to_Lcase = bert_layer.resolved_object.do_lower_case.numpy()
jetonizer = Bert_jeton(vocabulary_file, to_Lcase)

def jeton_tweet(tw_text):
    return jetonizer.convert_tokens_to_ids(jetonizer.tokenize(tw_text))
X_donnes=[]
Y_donnes=[]
for i in data:
    X_donnes.append(jeton_tweet(i))
for j in labels:
    Y_donnes.append(j)
mx_len = max([len(s) for s in X_donnes])
print(mx_len)
Xtrain = pad_sequences(X_donnes, maxlen=mx_len, padding='post')
print(Xtrain.shape)
print(len(Y_donnes))

from sklearn.model_selection import train_test_split


print("[INFO] Splitting donnees...")
trainX, testX, trainY, testY = train_test_split(Xtrain,Y_donnes, test_size=0.25, random_state=42)
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)
trainY=np.array(trainY)
vclen = len(jetonizer.vocab)




CnLS_neuro = Sequential()
CnLS_neuro.add(Embedding(vclen, 200, input_length=mx_len))
CnLS_neuro.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
CnLS_neuro.add(MaxPooling1D(pool_size=2))
CnLS_neuro.add(LSTM(100))
CnLS_neuro.add(Dense(2, activation='sigmoid'))
CnLS_neuro.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(CnLS_neuro.summary())
fpth="weights_best_cnn2.hLes_df5"
point_de  = ModelCheckpoint(fpth, monitor='val_acc', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
cbls = [point_de ]
CnLS_neuro.fit(trainX, trainY, epochs=20, batch_size=256,verbose = 1,callbacks = cbls,validation_data=(testX,testY))

model_json = CnLS_neuro.to_json()
with open("model_lstm201.json", "w") as json_file:
        json_file.write(model_json)
model.save_weights("lstm_weight201.h5")
print("[INFO] Saved model to disk")

## Prediction #################

def Robot_predict(twt):
    print(twt)
    X_data=[]
    twt=nettoyer_text(twt)
    twt=stemming(twt)
    print(twt)
    X_data.append(jeton_tweet(twt))
    print(X_data)
    test=pad_sequences(X_data, maxlen=68, padding='post')
    ypred=model.predict(test)
    res=np.argmax(ypred[0])
    print("res",res)
    if (res==1):
        return "Not a Bot Tweet"
    else:
        return "Bot Tweet"


##################### GUI ################################



from tkinter import *
from PIL import Image, ImageTk  
from Testbot import Robot_predict



def ajouter(a):
    global ma_liste
    ma_liste.insert(END, a)


def pr_get_val(val):
    print(val)
    
    output=Robot_predict(val)
    
    
    
    Bonjour .after(500, lambda : ajouter("Model loaded"))
    Bonjour .after(1700, lambda : ajouter("Text preprocessing"))
    Bonjour .after(2000, lambda : ajouter("Feature Extraction"))
    Bonjour .after(2500, lambda : ajouter("Prediction"))
    Bonjour .after(2800, lambda : ajouter("Result: "+output))
    Bonjour .after(3000, lambda : ajouter("============================"))
    Bonjour .after(3100, lambda :shrslt.config(text=output,fg="red"))
        
    
    
    
def domicile():
    global Bonjour , ma_liste,shrslt
    Bonjour  = Tk()
    Bonjour .geometry("1200x700+0+0")
    Bonjour .title("Home Page")

    picture = Image.open("twbot.jpg")
    picture = picture.resize((1200, 700), Image.ANTIALIAS) 
    pi = ImageTk.PhotoImage(picture)
    etiquette=Label(Bonjour ,image=pi,anchor=CENTER)
    etiquette.place(x=0,y=0)
  
    etiquette_info = Label(Bonjour , font=( 'aria' ,20, 'bold' ),text="CONTENT BASED BOT DETECTION",fg="white",bg="#000955",bd=10,anchor='w')
    etiquette_info.place(x=400,y=20)
    
    
    etiquette_info3 = Label(Bonjour , font=( 'aria' ,20 ),text="Enter Tweet ",fg="#000955",anchor='w')
    etiquette_info3.place(x=780,y=310)
    entree = Entry(Bonjour ,width=30,font="veranda 20")
    entree.place(x=650,y=360)
    ma_liste = Listbox(Bonjour ,width=50, height=20,bg="white")
    etiquette_info4 = Label(Bonjour , font=( 'aria' ,16 ),text="Process ",fg="#000955",anchor='w')
    etiquette_info4.place(x=180,y=270)

    ma_liste.place( x = 80, y = 300 )
    Bouton=Button(Bonjour ,padx=16,pady=8, bd=6 ,fg="white",font=('ariel' ,16,'bold'),width=10, text="Detect", bg="red",command=lambda:pr_get_val(entree.get()))
    Bouton.place(x=800, y=420)
    rslt = Label(Bonjour , font=( 'aria' ,20, ),text="RESULT :",fg="black",bg="white",anchor=W)
    rslt.place(x=640,y=580)
    shrslt = Label(Bonjour , font=( 'aria' ,20, ),text="",fg="blue",bg="white",anchor=W)
    shrslt.place(x=780,y=580)

   
    Bonjour .mainloop()


domicile()