#!/usr/bin/env python
# coding: utf-8

# LSTM Network to Generate West African folktales

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy

filename = 'data/input_afr.txt'
maxlen = 100
dataX = []
dataY = []

# load document and read text
def load_file(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

def create_mapping():
    # create mapping of unique chars to integers
    chars = sorted(list(set(doc)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    # summarize the loaded data
    n_chars = len(doc)
    n_vocab = len(chars)
    return n_chars, n_vocab, char_to_int

def prepare_dataset():
    # prepare the dataset of input to output pairs encoded as integers
    for i in range(0, n_chars - maxlen, 1):
        dataX.append([char_to_int[char] for char in doc[i:i + maxlen]])
        dataY.append(char_to_int[doc[i + maxlen]])
    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (len(dataX), maxlen, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)
    return X, y

def train_model():
    # define the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # save checkpoint if model has improved
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model
    model.fit(X, y, epochs=150, batch_size=64, callbacks=callbacks_list)
    return model

doc = load_file(filename)
n_chars, n_vocab, char_to_int = create_mapping()
X, y = prepare_dataset()
model = train_model()
model.save("chars_model.h5")
