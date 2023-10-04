
import os
import numpy as np
import pandas as pd
  

from six.moves import cPickle
#load vocabulary
save_dir = 'save1'

print("loading vocabulary...")
vocab_file = os.path.join(save_dir, "words_vocab.pkl")

with open(os.path.join(save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab, vocabulary_inv = cPickle.load(f)

vocab_size = len(words)

from keras.models import load_model
# load the model
print("loading model...")
model = load_model(save_dir + "/" + 'my_model_gen_sentences.06-5.83.hdf5')


def sample(preds_, temperature=1.0):
    
    preds_ = np.asarray(preds_).astype('float64')
    preds_ = np.log(preds_) / temperature
    exp_preds_ = np.exp(preds_)
    preds_ = exp_preds_ / np.sum(exp_preds_)
    probas = np.random.multinomial(1, preds_, 1)
    return np.argmax(probas)

seq_length = 30 
words_number = 30 # number of words to generate
def generate(sen):
    seed_sentences = sen #seed sentence to start the generating.

    #initiate sentences
    generated = ''
    sentence = []

    #we shate the seed accordingly to the neural netwrok needs:
    for i in range (seq_length):
        sentence.append("a")

    seed = seed_sentences.split()

    for i in range(len(seed)):
        sentence[seq_length-i-1]=seed[len(seed)-i-1]

    generated += ' '.join(sentence)

    #the, we generate the text
    for i in range(words_number):
        #create the vector
        x = np.zeros((1, seq_length, vocab_size))
        for t, word in enumerate(sentence):
            x[0, t, vocab[word]] = 1.

        #calculate next word
        preds_ = model.predict(x, verbose=0)[0]
        next_index = sample(preds_, 0.33)
        next_word = vocabulary_inv[next_index]

        #add the next word to the text
        generated += " " + next_word
        # shift the sentence by one, and and the next word at its end
        sentence = sentence[1:] + [next_word]
    senval=""
    cnt=0
    for i in range(len(generated)-2):
        
        if(generated[i]=='a' and generated[i+2]=='a'):
            cnt=i+1
        else:
            pass
    print(cnt)
    senval=generated[cnt+3:]
    
    print(senval)

    return senval
slist=[]
for i in words[:300]:
    try:
        senval=generate(i)
        slist.append(senval)
    except:
        pass


  
# Calling DataFrame constructor on list
# with indices and columns specified
df = pd.DataFrame(slist,columns =['text'])
print(df)
df.to_csv('out.csv',index=False)