import time
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout,GRU,Flatten
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pylab as plt

import plotter

def loadFile():
    trainFile = open('data_7000_new.csv', encoding="utf8")
    data = []
    for line in trainFile:
        line = line.replace('\n', '')
        columns = line.split(',')
        if columns[4] == 'funny' or columns[4] == 'very_funny' or columns[4] == 'hilarious':
            columns[4] = 1
        elif columns[4] == 'not_funny':
            columns[4] = 0

        if columns[5] == 'slight' or columns[5] == 'twisted_meaning' or columns[5] == 'very_twisted' or \
                columns[5] == 'general':
            columns[5] = 1
        elif columns[5] == 'not_sarcastic':
            columns[5] = 0

        if columns[6] == 'very_offensive' or columns[6] == 'slight' or columns[6] == 'hateful_offensive':
            columns[6] = 1
        elif columns[6] == 'not_offensive':
            columns[6] = 0
        if "<html>" not in columns[3]:
            data.append(columns)
    trainFile.close()
    txt = pd.DataFrame(data, columns=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    return txt

txt = loadFile()
def POSCleaning(txt):
    POSCleane = []
    for index, sentence in enumerate(txt[3]):
        tokenized = word_tokenize(sentence)
        tagges = nltk.pos_tag(tokenized)
        n_j_tag = []
        for tag in tagges:
            if tag[1] not in ['CC', 'CD','IN']:
                n_j_tag.append(tag[0])
        POSCleane.append(' '.join(n_j_tag))
    txt['POS'] = POSCleane
    return txt

def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    sentence = re.sub(r'^https?:\/\/.*[\r\n]*', '', sentence, flags=re.MULTILINE)

    return sentence


def loadData(txt):
    X = []
    for sen in txt.iterrows():
        X.append(preprocess_text(sen[1]['POS']))

    Y1 = txt[[4]]
    Y2 = txt[[5]]
    Y3 = txt[[6]]
    return np.array(X), np.array(Y1), np.array(Y2), np.array(Y3)



print("=========")
print("Loading File...")
txt = loadFile()
print("Cleaning By POS")
txt = POSCleaning(txt)

X, Y1, Y2, Y3 = loadData(txt)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y2, test_size=0.4, random_state=6)

maxlen = 20

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
# wordCoountPlotter(tokenizer)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
vocab_size = (len(tokenizer.word_index)+1)

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

m = np.mean(X_train)
v = np.std(X_train)
X_train = (X_train - m)/v
X_test = (X_test - m)/v
print('x_train shape:', X_train.shape)
# print('x_test shape:', X_test.shape)

from keras.utils import to_categorical


Y_train = np.array(Y_train)
print(Y_train)
Y_train = to_categorical(Y_train)
print(Y_train)
Y_test = np.array(Y_test)
Y_test = to_categorical(Y_test)
X_train = np.array(X_train)
X_test = np.array(X_test)

from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('./glove.twitter.27B.50d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 50))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
print('vect',embedding_vector)

#=============================

model = Sequential()
model.add(Embedding(vocab_size, 50,weights=[embedding_matrix],trainable=False,input_length=maxlen))
model.add(Bidirectional(LSTM(128,return_sequences=True)))
# model.add(Bidirectional(LSTM(2)))
model.add(LSTM(20))

model.add(Dropout(0.1))
model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(X_train, Y_train, batch_size=100
,epochs=30, verbose=1, validation_split=0.1)
print(model.summary())
import matplotlib.pyplot as plt
score = model.evaluate(X_test, Y_test, verbose=1,batch_size=100)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

from sklearn.metrics import precision_score,recall_score
y_true = Y_test
y_true = np.argmax(y_true, axis=1)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print('precision_score',precision_score(y_true, y_pred))
print('recall_score',recall_score(y_true, y_pred))


plotter.plot_acc_loss(history, score, 'test')

#
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# # "Loss"
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

