#!/usr/bin/env python
# coding: utf-8

# # An LSTM Model for Predicting Geographical Origin of Folktales.

from string import punctuation
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy import array
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout, Bidirectional
from keras.layers.core import Dense, SpatialDropout1D
from sklearn.metrics import roc_curve, auc
import pandas as pd 
import numpy as np
import keras

filename_afr = 'data/input_afr.txt'
filename_eur = 'data/input_eur.txt'
#Balanced or imbalanced dataset.
dataset = "imbalanced"
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 5000
# Max number of words in each story.
MAX_SEQUENCE_LENGTH = 500
# This is fixed.
EMBEDDING_DIM = 100
# Number of epochs.
epochs = 5

#Predict on new texts
stories = [('Foolish Anansi thought he could trick a fisherman into doing his work for him. "Let\'s go fishing," he suggested. "Very well," said the fisherman, who was clever and quite wise to Anansi\'s tricks. "I\'ll make the nets and you can get tired for me." "Wait," said Anansi, "I\'ll make the nets and you can get tired for me!" Anansi made nets as his friend pretended to be tired. They caught four fish. The fisherman said, "Anansi, you take these. I\'ll take tomorrow\'s catch. It might be bigger." Greedily imagining the next day\'s catch, Anansi said, "No, you take these and I\'ll take tomorrow\'s fish." But the next day, the nets were rotting away and no fish were caught.  The fisherman said, "Anansi, take these rotten nets to market. You can sell them for much money." When Anansi shouted, "Rotten nets for sale!" in the marketplace, people beat him with sticks. "Some partner you are," Anansi said to the fisherman as he rubbed his bruises.  "I took the beatings.  At least you could have taken the pain." Anansi never tried to trick the fisherman again!', 'West African')
,('Turtles used to live on the land, they say, until the time a clever turtle was caught by some hunters. They brought him to their village and placed the turtle before the Chief, who said, "How shall we cook him?" "You\'ll have to kill me first," said the turtle, "and take me out of this shell." "We\'ll break your shell with sticks," they said. "That\'ll never work," said the turtle, "Why don\'t you throw me in the water and drown me?!" "Excellent idea," said the Chief. They took the turtle to the river and threw him into the water to drown him. They were congratulating themselves on their success in drowning the turtle, when two little green eyes poked up in the water and the laughing turtle said, "Don\'t get those cooking pots out too fast, foolish people! As he swam away he said, "I think I\'ll spend most of my time from now on, safely in the water." It has been that way ever since!', 'West African')
,('A wolf, who was out searching for a meal, saw a goat feeding on grass on top of a high cliff. Wishing to get the goat to climb down from the rock and into his grasp, he called out to her. "Excuse me, dear Goat," he said in a friendly voice, "It is very dangerous for you to be at such a height. Do come down before you injure yourself. Besides, the grass is much greener and thicker down here. Take my advice, and please come down from that high cliff." But the goat knew too well of the wolf\'s intent. "You don\'t care if I injure myself or not. You don\'t care if I eat good grass or bad. What you care about is eating me."', 'Western European')
,('It was a windy afternoon - the trees sighing, the birds chirping, the frogs croaking, and the fowls cackling. There were clouds in the sky and empty tins and leaves were being blown about.', 'West African')
,('On a hill, a sheep that had no wool saw horses, one of them pulling a heavy wagon, one carrying a big load, and one carrying a man quickly. The sheep said to the horses: \"My heart pains me, seeing a man driving horses.\" The horses said: \"Listen, sheep, our hearts pain us when we see this: a man, the master, makes the wool of the sheep into a warm garment for himself. And the sheep has no wool.\" Having heard this, the sheep fled into the plain.', 'Western European')
,('When the animals heard this noise, they became very much afraid, and the started running away. They threw their tins away. Lion was in front and he ran very fast towards home.', 'West African')
,('"There is a stream about ten kilometres away." said Hare. "We can get some water from it. But there is a big Crab which frightens people who go there. If our strong brothers will try, we can get water presently."', 'West African')
,('Her parents did not think about it for long. "Birds of a feather, flock together," they thought, and gave their consent. So Fat Trina became Heinz\'s wife, and drove out both of the goats. Heinz now enjoyed life, having no work to rest from, but his own laziness. He went out with her only now and then, saying, "I\'m doing this so that afterwards I will enjoy resting more. Otherwise I shall lose all feeling for it."', 'Western European')
,('In later days two mighty swans have been seen to fly from the nest. A light shone far through the air, far over the lands of the earth; the swan, with the strong beating of his wings, scattered the twilight mists, and the starry sky was seen, and it was as if it came nearer to the earth. That was the swan Tycho Brahe. "Yes, then," you say; "but in our own days?"', 'Western European')
,('One fine day in winter some ants were busy drying their store of corn, which had got rather damp during a long spell of rain. Presently up came a grasshopper and begged them to spare her a few grains, "For," she said, "I\'m simply starving." The ants stopped work for a moment, though this was against their principles. "May we ask," said they, "what you were doing with yourself all last summer? Why didn\'t you collect a store of food for the winter?" "The fact is," replied the grasshopper, "I was so busy singing that I hadn\'t the time."', 'Western European')]


# load and read doc into memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc):
    list_tokens = []
    # split into tokens by white space
    tokens = doc.split('\n\n')
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    for story in tokens:
        story = story.translate(table)
        #print(story)
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in story.split() if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in story.split() if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in story.split() if len(word) > 1]
        list_tokens.append(tokens)
    return list_tokens

# load doc and add to vocab
def preprocess_stories(filename):
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)   
    return tokens

def run_model():
    global auc
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=epochs, batch_size=64,validation_split=0.1)
    loss, acc = model.evaluate(X_test, Y_test, verbose=2)
    y_true = pd.Series(Y_test.columns[np.where(Y_test!=0)[1]])
    y_predict = model.predict(X_test).round()
    y_pred = []
    for i in y_predict:
        y_pred.append(np.argmax(i))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = auc(fpr, tpr)
    print('Test Accuracy: %f' % (acc), 'Test AUC-score: %f' % (auc), "\n")
    return model

def predict_background(story):
    text = np.array([story])
    tk = keras.preprocessing.text.Tokenizer( MAX_NB_WORDS,split=" ")
    tk.fit_on_texts(text)
    seq = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    labels = ['West African','Western European']
    prediction = model.predict(np.array(padded))
    return labels[np.argmax(prediction)]

afr_stories = preprocess_stories(filename_afr)
eur_stories = preprocess_stories(filename_eur)

if dataset == "balanced":
    eur_stories = eur_stories[:len(afr_stories)]
elif dataset == "imbalanced":
    pass

all_stories = afr_stories + eur_stories
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts([story for story in all_stories])
word_index = tokenizer.word_index
X = tokenizer.texts_to_sequences([story for story in all_stories])
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
y = array([0 for _ in range(len(afr_stories))] + [1 for _ in range(len(eur_stories))])
Y = pd.get_dummies(y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10, random_state = 42)

model = run_model()

for story in stories:
    prediction = predict_background(story[0])
    if story[1] == prediction:
        print(prediction, "- Correct!", "\n")
    else:
        print(prediction, "- Wrong!", "\n")

